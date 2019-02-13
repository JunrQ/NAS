import torch
import torch.nn as nn

from collections import OrderedDict

# SENet
# See https://github.com/moskomule/senet.pytorch/tree/master/senet
class SELayer(nn.Module):
  def __init__(self, channel, reduction=16):
    super(SELayer, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid())
  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, stride=1, 
               reduction=16):
    super(SEBasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes, 1)
    self.bn2 = nn.BatchNorm2d(planes)
    self.se = SELayer(planes, reduction)
    downsample = None
    if stride != 1 or inplanes != planes:
      downsample = nn.Sequential(
          nn.Conv2d(inplanes, planes, kernel_size=1, 
                    stride=stride, bias=False),
          nn.BatchNorm2d(planes))
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.se(out)
    if self.downsample is not None:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

# ResNet
# See https://github.com/aaron-xichen/pytorch-playground/blob/master/imagenet/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1):
  # "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, groups=groups,
                   stride=stride, padding=1, bias=False)

class ResNetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, stride=1, groups=1, shuffle=False,
               expansion=1):
    super(ResNetBasicBlock, self).__init__()
    m = OrderedDict()
    m['conv1'] = conv3x3(inplanes, planes * expansion, stride) # groups=groups
    # if shuffle and groups > 1:
    #   m['shuffle1'] = ChannelShuffle(groups)
    m['bn1'] = nn.BatchNorm2d(planes)
    m['relu1'] = nn.ReLU(inplace=True)
    m['conv2'] = conv3x3(planes * expansion, planes, groups=groups)
    if shuffle and groups > 1:
      m['shuffle2'] = ChannelShuffle(groups)
    m['bn2'] = nn.BatchNorm2d(planes)
    self.group1 = nn.Sequential(m)
    self.relu= nn.Sequential(nn.ReLU(inplace=True))
    downsample = None
    if stride != 1 or inplanes != planes:
      downsample = nn.Sequential(
          nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes))
    self.downsample = downsample
  def forward(self, x):
    if self.downsample is not None:
        residual = self.downsample(x)
    else:
        residual = x
    out = self.group1(x) + residual
    out = self.relu(out)
    return out

class ResNetBottleneck(nn.Module):
  def __init__(self, inplanes, planes, stride=1, groups=1, shuffle=False,
               expansion=1):
    super(ResNetBottleneck, self).__init__()
    m  = OrderedDict()
    r = int(planes * 0.25 * expansion)
    m['conv1'] = nn.Conv2d(inplanes, r, kernel_size=1, bias=False)
    m['bn1'] = nn.BatchNorm2d(r)
    m['relu1'] = nn.ReLU(inplace=True)
    m['conv2'] = nn.Conv2d(r, r, kernel_size=3, stride=stride, padding=1, bias=False)
    m['bn2'] = nn.BatchNorm2d(r)
    m['relu2'] = nn.ReLU(inplace=True)
    m['conv3'] = nn.Conv2d(r, planes, groups=groups, kernel_size=1, bias=False)
    if shuffle and groups > 1:
      m['shuffle2'] = ChannelShuffle(groups)
    m['bn3'] = nn.BatchNorm2d(planes)
    self.group1 = nn.Sequential(m)
    self.relu= nn.Sequential(nn.ReLU(inplace=True))
    downsample = None
    if stride != 1 or inplanes != planes:
      downsample = nn.Sequential(
          nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes))
    self.downsample = downsample

  def forward(self, x):
    if self.downsample is not None:
        residual = self.downsample(x)
    else:
        residual = x
    out = self.group1(x) + residual
    out = self.relu(out)
    return out

class ChannelShuffle(nn.Module):
  def __init__(self, group=1):
    assert group > 1
    super(ChannelShuffle, self).__init__()
    self.group = group
  def forward(self, x):
    """https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % self.group == 0)
    channels_per_group = num_channels // self.group
    # reshape
    x = x.view(batchsize, self.group, channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()
  def forward(self, x):
    return x

def get_blocks(cifar10=False, face=False):
  BLOCKS = []
  _f = [64, 256, 512, 1024, 2048]
  _n = [3, 4, 6, 3]
  if cifar10:
    assert False
  elif face:
    assert not cifar10
  else:
    assert False
  _group = [1, 4, 1, 4, 1]
  BLOCKS.append(nn.Conv2d(3, _f[0], 3, 1, padding=1))
  
  c_in = _f[0]
  for n_idx in range(len(_n)):
    c_out = _f[n_idx + 1]
    stride = 2

    for inner_idx in range(_n[n_idx]):
      tmp_block = []

      for b_idx in range(len(_group)):
        group = _group[b_idx]

        if b_idx < 2:
          tmp_block.append(ResNetBasicBlock(c_in, c_out,
                            stride=stride, groups=group,
                            shuffle=True))
        elif b_idx >=2 and b_idx < 4:
          tmp_block.append(ResNetBottleneck(c_in, c_out,
                            stride=stride, groups=group,
                            shuffle=True))
        else:
          tmp_block.append(SEBasicBlock(c_in, c_out, stride))

      if inner_idx > 0 and ((c_in == c_out) and (stride == 1)):
        tmp_block.append(Identity())

      BLOCKS.append(tmp_block)
      stride = 1
      c_in = c_out
  BLOCKS.append(nn.Conv2d(c_out, 192, 1, padding=0))
  assert len(BLOCKS) == 18
  return BLOCKS
