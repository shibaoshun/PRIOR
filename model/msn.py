import torch
from torch import nn
import torch.nn.functional as F
from .fftformer_arch import TransformerBlock

def rot180(d):  # 转置
    ###  the filtersize must be a 奇数
    d = d.permute(1, 0, 2, 3)
    filtersize = d.shape[3]
    a = d.clone()
    for i in range(1, (filtersize + 1)):
        a[:, :, (i - 1), :] = d[:, :, (filtersize - i), :]
    c = a.clone()
    for i in range(1, (filtersize + 1)):
        c[:, :, :, (i - 1)] = a[:, :, :, (filtersize - i)]
    return c

class SoftThreshold(nn.Module):  # 软阈值函数
    def __init__(self):
        super(SoftThreshold, self).__init__()

    def forward(self, x, threshold):
        mask1 = (x > threshold).float()
        mask2 = (x < -threshold).float()
        out = mask1.float() * (x - threshold)
        out += mask2.float() * (x + threshold)
        return out

class WeiNet(nn.Module):
    def __init__(self, featuremap):
        super(WeiNet, self).__init__()
        ############### for x
        featuremapc=81
        self.apply_A = torch.nn.Conv2d(featuremap, featuremapc, kernel_size=3, stride=1, padding=1)  # 81
        self.apply_B = torch.nn.Conv2d(featuremapc, featuremapc, kernel_size=3, stride=1, padding=1)
        self.apply_C = torch.nn.Conv2d(featuremapc, featuremapc, kernel_size=3, stride=1, padding=1)
        self.apply_D = torch.nn.Conv2d(featuremapc, featuremapc, kernel_size=3, stride=1, padding=1)
        self.apply_E = torch.nn.Conv2d(featuremapc, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ############# conv +  relu
        temp1 = self.relu(self.apply_A(x))
        temp2 = self.relu(self.apply_B(temp1))
        temp3 = self.relu(self.apply_C(temp2))
        temp4 = self.relu(self.apply_D(temp3))
        y = self.apply_E(temp4)
        return y

class F_ext(nn.Module):
    def __init__(self, in_nc=2, nf=81):
        super(F_ext, self).__init__()
        stride = 2
        pad = 0
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.act(self.conv1(self.pad(x)))
        conv2_out = self.act(self.conv2(self.pad(conv1_out)))
        conv3_out = self.act(self.conv3(self.pad(conv2_out)))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)   #zheli  输出(1,81)

        return out

class MSN(nn.Module):
    def __init__(self,args):
        super(MSN, self).__init__()

        self.kernel_size1 = 3
        self.num_filters1 = self.kernel_size1 ** 2
        self.conv_w1 = nn.Conv2d(1, self.num_filters1, self.kernel_size1, stride=1, padding=0, bias=False)
        self.conv_c1 = nn.ConvTranspose2d(25, 9, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.soft_threshold = SoftThreshold()
        self.rpad1 = nn.ReflectionPad2d(self.kernel_size1 // 2)  #怎么padding不影响，试试

        self.kernel_size2 =5
        self.num_filters2 = self.kernel_size2 ** 2
        self.conv_w2 = nn.Conv2d(1, self.num_filters2, self.kernel_size2, stride=2, padding=0, bias=False)
        self.conv_c2 = nn.ConvTranspose2d(49, 25, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.rpad2 = nn.ReflectionPad2d(self.kernel_size2 // 2)

        self.kernel_size3 = 7
        self.num_filters3 = self.kernel_size3 ** 2
        self.conv_w3 = nn.Conv2d(1, self.num_filters3, self.kernel_size3, stride=4, padding=0, bias=False)
        self.conv_c3 = nn.Conv2d(9, 49, kernel_size=3, stride=4, padding=0, bias=False)
        self.rpad3 = nn.ReflectionPad2d(self.kernel_size3 // 2)

        self.tcnet = nn.ModuleList()
        for i in range(args.layers):
            self.tcnet.append(TransformerBlock(dim=9))  # self.kernel_size1    #15
            self.tcnet.append(TransformerBlock(dim=25))  # self.kernel_size2    #40
            self.tcnet.append(TransformerBlock(dim=49))  # self.kernel_size3    #65

        self.wnet = nn.ModuleList()
        for i in range(args.layers):
            self.wnet.append(WeiNet(3))
            self.wnet.append(WeiNet(3))

        self.F_ext_net = F_ext(in_nc=1, nf=81)


    def forward(self, x, i, sr):
        Wx11 = self.conv_w1(self.rpad1(x))
        Wx22 = self.conv_w2(self.rpad2(x))
        Wx33 = self.conv_w3(self.rpad3(x))

        Wx21 = self.conv_c1(Wx22)
        Wx32 = self.conv_c2(Wx33)
        Wx13 = self.conv_c3(Wx11)

        xsr = sr * torch.ones([1,1,512,512]).cuda()
        prompt = self.F_ext_net(xsr)

        epsilon1,p1 = self.tcnet[3 * i](Wx21,Wx11,prompt)
        z1 = self.soft_threshold(Wx11, epsilon1)
        weightT1 = rot180(self.conv_w1.weight)
        Wt1 = F.conv2d(self.rpad1(z1), weightT1, stride=1, padding=0) / (self.kernel_size1 ** 2)

        epsilon2,p2 = self.tcnet[3 * i + 1](Wx32,Wx22,prompt)
        z2 = self.soft_threshold(Wx22, epsilon2)
        weightT2 = rot180(self.conv_w2.weight)
        weightT2 = torch.transpose(weightT2, 0, 1)
        Wt2 = F.conv_transpose2d(z2, weightT2, stride=2, padding=2, output_padding=1) / (self.kernel_size2 ** 2)

        epsilon3,p3 = self.tcnet[3 * i + 2](Wx13,Wx33,prompt)
        z3 = self.soft_threshold(Wx33, epsilon3)
        weightT3 = rot180(self.conv_w3.weight)
        weightT3 = torch.transpose(weightT3, 0, 1)
        Wt3 = F.conv_transpose2d(z3, weightT3, stride=4, padding=3, output_padding=3) / (self.kernel_size3 ** 2)

        feat1 = Wt1
        feat2 = Wt2
        feat3 = Wt3

        feat = torch.cat((feat1, feat2, feat3), dim=1)
        w1 = self.wnet[2 * i](feat)
        w1 = torch.clip(w1, 0, 1)
        w2 = self.wnet[2 * i+ 1](feat)
        w2 = torch.clip(w2, 0, 1)
        w3 = 1-w1-w2
        w3 = torch.clip(w3, 0, 1)

        out = feat1*w1 + feat2*w2 + feat3*w3

        return out,self.conv_w1.weight,self.conv_w2.weight,self.conv_w3.weight, p1, p2, p3
