# self-attention GAN: Han Zhang et al - https://arxiv.org/pdf/1805.08318.pdf

import torch 
import torch.nn as nn  
import torch.nn.functional as F


def specNorm_conv2d(eps=1e-12, **kwargs):
    cnn = nn.Conv2d(**kwargs)
    o = nn.utils.spectral_norm(cnn, eps=eps)
    return o

def specNorm_linear(eps=1e-12, **kwargs):
    linear = nn.Linear(**kwargs)
    o = nn.utils.spectral_norm(linear, eps=eps)
    return o

def specNorm_embedding(eps=1e-12, **kwargs):
    emb = nn.Embedding(**kwargs)
    o = nn.utils.spectral_norm(emb, eps=eps)
    return o


class Attention_block(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_f = specNorm_conv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = specNorm_conv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_h = specNorm_conv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = specNorm_conv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # f(x)
        f_x = self.snconv1x1_f(x)
        f_x = f_x.view(-1, ch//8, h*w)
        # g(x)
        g_x = self.snconv1x1_g(x)
        g_x = self.maxpool(phi)
        g_x = g_x.view(-1, ch//8, h*w//4)
        # Attention map
        attn = torch.bmm(f_x.permute(0, 2, 1), g_x)
        attn = self.softmax(attn)
        # h(x) 
        h_x = self.snconv1x1_h(x)
        h_x = self.maxpool(h_x)
        h_x = h_x.view(-1, ch//2, h*w//4)

        # Attn_g - o_conv
        attn_g = torch.bmm(h_x, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma*attn_g
        return out

