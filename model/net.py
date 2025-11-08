''' Genereal network structure with Dn-CNN, Db-INV and P-FCN '''
import torch
from torch import nn
import numpy as np
import odl
from odl.contrib import torch as odl_torch

from .msn import MSN

import matplotlib.pyplot as plt
import cv2



def radon_transform():
    xx = 200
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
    detectors = np.array(800).astype(int)
    detector_partition = odl.uniform_partition(-480, 480, detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,det_radius=290)  # FanBeamGeometry
    operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm = odl.operator.power_method_opnorm(operator)
    op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

    op_layer = odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
    op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

proj, back_proj, _, op_norm = radon_transform()

def radon(img, sr):
    if len(img.shape) == 4:
        img = img.squeeze(1)
    sino = proj(img)
    if sr == 6:
        sino[:, 1:361:6, :] = 0
        sino[:, 2:362:6, :] = 0
        sino[:, 3:363:6, :] = 0
        sino[:, 4:364:6, :] = 0
        sino[:, 5:365:6, :] = 0
    elif sr == 4:
        sino[:, 1:361:4, :] = 0
        sino[:, 2:362:4, :] = 0
        sino[:, 3:363:4, :] = 0
    elif sr == 3:
        sino[:, 1:361:3, :] = 0
        sino[:, 2:362:3, :] = 0
    elif sr == 2:
        sino[:, 1:361:2, :] = 0
    elif sr == 5:
        sino[:, 1:361:5, :] = 0
        sino[:, 2:362:5, :] = 0
        sino[:, 3:363:5, :] = 0
        sino[:, 4:364:5, :] = 0
    elif sr == 8:
        sino[:, 1:361:8, :] = 0
        sino[:, 2:362:8, :] = 0
        sino[:, 3:363:8, :] = 0
        sino[:, 4:364:8, :] = 0
        sino[:, 5:365:8, :] = 0
        sino[:, 6:366:8, :] = 0
        sino[:, 7:367:8, :] = 0
    #sinosv = sino[:, 0:360:3, :]
    return sino.unsqueeze(1)


def iradon(sino):
    if len(sino.shape) == 4:
        sino = sino.squeeze(1)
    #sino360 = buling(sino).cuda()
    img = back_proj(sino)   #back_proj(sino / op_norm)
    return img.unsqueeze(1)

###重型网络（UNet）
class Gradient_Descent(nn.Module):
    def __init__(self, args):
        super(Gradient_Descent,self).__init__()
        self.args = args
        self.etaconst = torch.tensor(self.args.eta).float()
        self.tauconst = torch.tensor(self.args.tau).float()
        self.eta = nn.Parameter(data=self.etaconst, requires_grad=True)
        self.tau = nn.Parameter(data=self.tauconst, requires_grad=True)
        self.CNN = MSN(args)

    def forward(self, sino, fbpu, sr ):
        x = fbpu
        for i in range(self.args.layers):
            res = sino - radon(x, sr)
            grad1 = iradon(res)
            tfx,W1,W2,W3, p1, p2, p3=self.CNN(x, i, sr)

            x = x + self.eta * grad1 - self.tau * tfx

        return x, W1, W2, W3, p2
