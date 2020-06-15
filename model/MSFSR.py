import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np


###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)

def BUM(norm_layer, scale=8):
    Basic_Upscale_Module = [nn.Conv2d(3, 64, kernel_size=5, padding=5 // 2),
                            norm_layer(64),
                            nn.ReLU(True),
                            RCAB(default_conv, 64, 3, 16, bias=True, bn=False),
                            nn.Conv2d(64, 32, kernel_size=3, padding=3 // 2),
                            norm_layer(32),
                            nn.ReLU(True),
                            nn.Conv2d(32, 3 * (scale ** 2), kernel_size=3, padding=3 // 2),
                            nn.PixelShuffle(scale)]
    Basic_Upscale_Module = nn.Sequential(*Basic_Upscale_Module)
    return Basic_Upscale_Module

def BEM(norm_layer):
    Boundary_Estimation_Network = [nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                norm_layer(64),
                                nn.ReLU(True)]
    Boundary_Estimation_Network += [Residual(64, 128)]
    for i in range(2):
        Boundary_Estimation_Network += [ResnetBlock(128, 'reflect', norm_layer, False, False)]
    for i in range(2):
        Boundary_Estimation_Network += [HourGlassBlock(128, 3, norm_layer)]

    Boundary_Estimation_Network += [nn.Conv2d(128, 11, kernel_size=1, stride=1),
                                    norm_layer(11),
                                    nn.ReLU(True)]
    Boundary_Estimation_Network = nn.Sequential(*Boundary_Estimation_Network)
    return Boundary_Estimation_Network

def BFM_head(norm_layer):
    BFM_head = [nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(True)]
    BFM_head += [ResnetBlock(64, 'reflect', norm_layer, False, False),
                            SELayer(64, reduction=8)]

    BFM_head += [nn.ReflectionPad2d(1),
                 nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
                 nn.Tanh(),
                 SELayer(64,reduction=8)
                 ]
    BFM_head = nn.Sequential(*BFM_head)

    return BFM_head

def BFM_Tail(norm_layer):
    BFM_Tail = [nn.Conv2d(64 + 11, 64, kernel_size=3,stride=1,padding=1,bias=False),
                norm_layer(64),
                nn.ReLU(True),
                SELayer(64,reduction=8)]

    BFM_Tail += [nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(True)]
    for i in range(1):
        BFM_Tail += [ResnetBlock(64, 'reflect', norm_layer, False, False),
                            SELayer(64, reduction=8)]

    BFM_Tail += [nn.ReflectionPad2d(1),
                 nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0, bias=False),
                 nn.Tanh(),
                 ]
    BFM_Tail = nn.Sequential(*BFM_Tail)
    return BFM_Tail

def defineThreeStageGenerator(input_nc, output_nc, ngf=64, norm='batch', use_dropout=False, gpu_ids=[], use_parallel=True,
                            learn_residual=False, upscale=8):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    netG = Generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                     gpu_ids=gpu_ids, use_parallel=use_parallel, learn_residual=learn_residual)
    if use_gpu:
        assert (torch.cuda.is_available())

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, norm_layer=nn.BatchNorm2d,
                 gpu_ids=[], use_parallel=False, learn_residual=False, padding_type='reflect', scale=2):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual
        self.scale = scale

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.BUM = BUM(norm_layer, self.scale)
        self.BFM_head = BFM_head(norm_layer)
        self.BEM = BEM(norm_layer)
        self.BFM_Tail = BFM_Tail(norm_layer)

    def forward(self, input, is_hr=False):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor) and self.use_parallel:
            if is_hr == True:
                heatmaps = nn.parallel.data_parallel(self.prior_Estimation_Network, input, self.gpu_ids)
                return heatmaps
            else:
                out1 = nn.parallel.data_parallel(self.BUM, input, self.gpu_ids)
                Hidden_Representations = nn.parallel.data_parallel(self.BFM_head, out1, self.gpu_ids)
                heatmaps = nn.parallel.data_parallel(self.BEM, out1, self.gpu_ids)
                concatenation = torch.cat((Hidden_Representations, heatmaps), 1)
                output = nn.parallel.data_parallel(self.BFM_Tail, concatenation, self.gpu_ids)
        else:
            if is_hr == True:
                heatmaps = self.BEM(input)
                return heatmaps
            else:
                out1 = self.BUM(input)
                Hidden_Representations = self.BFM_head(out1)
                heatmaps = self.BEM(out1)
                concatenation = torch.cat((Hidden_Representations, heatmaps), 1)
                output = self.BFM_Tail(concatenation)

        if self.learn_residual:
            output = out1 + output
            output = torch.clamp(output, min=-1, max=1)
        return out1, heatmaps, output

class HourGlassBlock(nn.Module):
    def __init__(self, dim, n, norm_layer):
        super(HourGlassBlock, self).__init__()
        self._dim = dim
        self._n = n
        self._norm_layer = norm_layer
        self._init_layers(self._dim, self._n, self._norm_layer)

    def _init_layers(self, dim, n, norm_layer):
        setattr(self, 'res' + str(n) + '_1', Residual(dim, dim))
        setattr(self, 'pool' + str(n) + '_1', nn.MaxPool2d(2, 2))
        setattr(self, 'res' + str(n) + '_2', Residual(dim, dim))
        if n > 1:
            self._init_layers(dim, n - 1, norm_layer)
        else:
            self.res_center = Residual(dim, dim)
        setattr(self, 'res' + str(n) + '_3', Residual(dim, dim))
        setattr(self, 'unsample' + str(n), nn.Upsample(scale_factor=2))

    def _forward(self, x, dim, n):
        up1 = x
        up1 = eval('self.res' + str(n) + '_1')(up1)
        low1 = eval('self.pool' + str(n) + '_1')(x)
        low1 = eval('self.res' + str(n) + '_2')(low1)
        if n > 1:
            low2 = self._forward(low1, dim, n - 1)
        else:
            low2 = self.res_center(low1)
        low3 = low2
        low3 = eval('self.' + 'res' + str(n) + '_3')(low3)
        up2 = eval('self.' + 'unsample' + str(n)).forward(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._forward(x, self._dim, self._n)

# class SELayer(nn.Module):
# 	def __init__(self,channel,reduction = 16):
# 		super(SELayer,self).__init__()
# 		self.avg_pool = nn.AdaptiveAvgPool2d(1)
# 		self.fc = nn.Sequential(
# 			nn.Linear(channel, channel//reduction,bias = False),
# 			nn.ReLU(inplace = True),
# 			nn.Linear(channel // reduction, channel, bias = False),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self,x):
# 		b,c,_,_ = x.size()
# 		y = self.avg_pool(x).view(b,c)
# 		y = self.fc(y).view(b,c,1,1)
# 		return x*y.expand_as(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Residual(nn.Module):
    def __init__(self, ins, outs):
        super(Residual, self).__init__()
        self.convBlock = nn.Sequential(
            nn.BatchNorm2d(ins),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins, outs // 2, 1),
            nn.BatchNorm2d(outs // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs // 2, outs // 2, 3, 1, 1),
            nn.BatchNorm2d(outs // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outs // 2, outs, 1)
        )
        if ins != outs:
            self.skipConv = nn.Conv2d(ins, outs, 1)
        self.ins = ins
        self.outs = outs

    def forward(self, x):
        residual = x
        x = self.convBlock(x)
        if self.ins != self.outs:
            residual = self.skipConv(residual)
        x += residual
        return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, updimension)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, updimension):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        in_chan = dim
        if updimension == True:
            out_chan = in_chan * 2
        else:
            out_chan = dim
        conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == "__main__":
    network = defineOneStageGenerator(input_nc=3, output_nc=3)
    print(network)
