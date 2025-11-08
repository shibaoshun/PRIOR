import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)  #zhege

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x1,x2):
        k = x1
        hidden = self.to_hidden(x2)
        q, v = self.to_hidden_dw(hidden).chunk(2, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        self.attn = FSAS(dim, bias)

        self.squeeze = nn.Linear(81, dim, bias=True)
        self.prompt_scale1 = nn.Linear(dim, dim, bias=True)
        self.prompt_shift1 = nn.Linear(dim, dim, bias=True)

        self.apply_E = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.apply_F = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.apply_G = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dim =dim


    def forward(self, x1, x2, prompt):

        prompts = self.squeeze(prompt)
        scale1 = self.prompt_scale1(prompts)
        shift1 = self.prompt_shift1(prompts)

        xmc = x2 + self.attn(self.norm1(x1), self.norm2(x2))

        temp1 = self.relu(self.apply_E(xmc))
        temp1p1 = temp1 * scale1.view(-1, self.dim, 1, 1)
        temp1p = temp1p1 + shift1.view(-1, self.dim, 1, 1) + temp1
        temp2 = self.relu(self.apply_F(temp1p))
        out = self.apply_G(temp2)

        pout = torch.mean(temp1p1, dim=1, keepdim=True)
        pout = rearrange(pout, 'b c h w -> b (c h w)')

        return out,pout