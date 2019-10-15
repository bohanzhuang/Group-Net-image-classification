import torch
import torch.nn as nn
import numpy as np 
from torch.autograd.function import Function
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Q_A(torch.autograd.Function):  #dorefanet, but constrain to {-1, 1}
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)    
        return x.sign()                     
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input>1.0, 0.0)
        grad_input.masked_fill_(input<-1.0, 0.0)
        mask_pos = (input>=0.0) & (input<1.0)
        mask_neg = (input<0.0) & (input>=-1.0)
        grad_input.masked_scatter_(mask_pos, input[mask_pos].mul_(-2.0).add_(2.0)) 
        grad_input.masked_scatter_(mask_neg, input[mask_neg].mul_(2.0).add_(2.0)) 
        return grad_input * grad_output



class Q_W(torch.autograd.Function):  # xnor-net, but gradient use identity approximation
    @staticmethod
    def forward(ctx, x):
        return x.sign() * x.abs().mean()
    @staticmethod
    def backward(ctx, grad):
        return grad


def quantize_a(x):
    x = Q_A.apply(x)
    return x


def quantize_w(x):
    x = Q_W.apply(x)
    return x


def fw(x, bitW):
    if bitW == 32:
        return x
    x = quantize_w(x)
    return x


def fa(x, bitA):
    if bitA == 32:
        return x
    return quantize_a(x)


def nonlinear(x):
    return torch.clamp(torch.clamp(x, max=1.0), min=0.0)



class self_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, bitW, kernel_size, stride=1, padding=0, bias=False):
        super(self_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bitW = bitW
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)  
            input = F.pad(input, padding_shape, 'constant', 1)       #padding 1
        output = F.conv2d(input, fw(self.weight, self.bitW), bias=self.bias, stride=self.stride, dilation=self.dilation, groups=self.groups)
        return output



class new_conv(nn.Module):
    def __init__(self, num_bases, bitW, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(new_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.convs = nn.ModuleList([self_conv(self.in_channels, self.out_channels, bitW, self.kernel_size, self.stride, bias=self.bias) for i in range(num_bases)])
        self.scales = nn.ParameterList([nn.Parameter(torch.rand(1).cuda(), requires_grad=True) for i in range(num_bases)])


    def forward(self, input):
        output = None
        if self.padding > 0:
            padding_shape = (self.padding, self.padding, self.padding, self.padding)  
            input = F.pad(input, padding_shape, 'constant', 1)       #padding 1

        for scale, module in zip(self.scales, self.convs):
            if output is None:
                output = scale * module(input)
            else:
                output += scale * module(input)
        return output



class clip_nonlinear(nn.Module):
    def __init__(self, bitA):
        super(clip_nonlinear, self).__init__()
        self.bitA = bitA

    def forward(self, input):
        return fa(input, self.bitA)

