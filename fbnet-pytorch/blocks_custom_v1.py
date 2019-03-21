import torch
import torch.nn as nn

from collections import OrderedDict
from blocks_custom import ResNetBasicBlock, ResNetBottleneck, Identity
from shufflenet_v2 import ShuffleV2BasicBlock

def get_blocks(cifar10=False, face=False):
  BLOCKS = []
  _f = [64, 128, 256, 512, 1024]
  _n = [2, 2, 2, 2]
  if cifar10:
    # assert False
    pass
  elif face:
    assert not cifar10
  else:
    assert False
  _group = [1, 2, 4, 1, 1]
  BLOCKS.append(nn.Conv2d(3, _f[0], 3, 1, padding=1))
  
  c_in = _f[0]
  for n_idx in range(len(_n)):
    c_out = _f[n_idx + 1]
    stride = 2

    for inner_idx in range(_n[n_idx]):
      tmp_block = []

      for b_idx in range(len(_group)):
        group = _group[b_idx]

        name = 'tbs_%d_%d_%d' % (n_idx, inner_idx, b_idx)

        if b_idx < 3:
          tmp_block.append(ResNetBasicBlock(c_in, c_out,
                            stride=stride, groups=group,
                            shuffle=True))
        elif b_idx == 3:
          tmp_block.append(ResNetBottleneck(c_in, c_out,
                            stride=stride, groups=group,
                            shuffle=True))
        else:
          tmp_block.append(ShuffleV2BasicBlock(name, c_in, c_out, stride, 1))

      if inner_idx > 0 and ((c_in == c_out) and (stride == 1)):
        tmp_block.append(Identity())

      BLOCKS.append(tmp_block)
      stride = 1
      c_in = c_out
  BLOCKS.append(nn.Conv2d(c_out, 192, 1, padding=0))
  # BLOCKS.append(nn.BatchNorm2d(192))
  assert len(BLOCKS) == 10
  return BLOCKS
