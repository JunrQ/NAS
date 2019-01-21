import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import sys
import logging
from time import gmtime, strftime
# mxnet
sys.path.insert(0, '/home/zhouchangqing/mxnet/incubator-mxnet_12_26/python')
import mxnet as mx
import numpy as np
import math
import argparse
import time

from blocks import block_factory
from blocks_se import block_factory_se

times = 5000
_e = [1, 1, 3, 6,
      1, 1, 3, 6]
_kernel = [3, 3, 3, 3,
          5, 5, 5, 5]
_group = [1, 2, 1, 1,
          1, 2, 1, 1]

logging.basicConfig(level=logging.INFO)

nchannel = 3
def speed_test(input_shape, s_size, num_filter, ctx = mx.gpu(0)):
  input_channels = input_shape[0]
  data = mx.sym.var('data')
  block_list = []
  for block_idx in range(8):
    kernel_size = (_kernel[block_idx], _kernel[block_idx])
    group = _group[block_idx]
    expansion = _e[block_idx]
    stride = (s_size, s_size)
    prefix = "block_%d" % block_idx

    block_out = block_factory(data, input_channels=input_channels,
                        num_filters=num_filter, kernel_size=kernel_size,
                        prefix=prefix, expansion=expansion,
                        group=group, shuffle=False,
                        stride=stride, bn=False)
    # block_out = mx.sym.BatchNorm(data=block_out, fix_gamma=False, eps=2e-5, momentum=0.9)
    if (input_channels == num_filter) and (s_size == 1):
      block_out = block_out + data
    block_out = mx.sym.expand_dims(block_out, axis=1)
    block_list.append(block_out)
  block_list.append(data)
  # print(block_list)

  speed_list = []
  for i, sym in enumerate(block_list):

    mod = mx.mod.Module(symbol=sym, context=[ctx], data_names=['data'], label_names=None)
    mod.bind(data_shapes=[['data', (1, ) + input_shape]], for_training=False)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2), 
                      allow_missing=True, allow_extra=True)

    data = mx.nd.random.normal(shape=(1, ) + input_shape, ctx=ctx)
    _dataiter = mx.io.NDArrayIter(data={'data': data},
                                  batch_size=1)
    tmp_data = _dataiter.next()
    mod.forward(tmp_data)
    mod.get_outputs()[0].asnumpy()
    # tic
    start = time.time()
    for _ in range(times):
      mod.forward(tmp_data)
      y = mod.get_outputs()
      y[0].asnumpy()
    # toe
    end = time.time()
    speed = 1.0 * (end - start) / times * 1000
    speed_list.append(speed)
    msg = "Block %d speed %f" % (i, speed)
    print(msg)
  print(' '.join([str(t) for t in speed_list]))


_unistage = 4

_n = [3, 1, 1, 2]
_f = [64, 256, 512, 1024, 2048]

_se = [0, 1, 0, 1,
       0, 1, 0, 1]
_k = [3, 3, 3, 3,
      3, 3, 3, 3]
_g = [1, 1, 2, 2,
      1, 1, 2, 2]

_n = [3,4,6,3]
_f = [256, 512, 1024, 2048]
_bottle_neck = [1,1,0,0,0]
_se =  [0, 0, 0, 1, 0]
_k =   [3, 3, 3, 3, 3]
_g =   [1, 2, 1, 1, 2]

def speed_test_se(input_shape, stride, num_filter, dim_match = False,deform_conv= False,ctx = mx.gpu(1)):

  data = mx.sym.var('data')
  block_list = []
  group_list = []
  se_list = []

  for block_idx in range(len(_g)):
    kernel_size = _k[block_idx]

    group = _g[block_idx]
    se = _se[block_idx]
    stride = stride

    prefix = "block_%d" % block_idx
    type = 'bottle_neck' if block_idx<=1 else 'resnet'
    
    block_out = block_factory_se(input_symbol=data, name=prefix, num_filter=num_filter, group=group, stride=stride,
                                 se=se, k_size=kernel_size, type=type,dim_match=dim_match)

    block_out = mx.sym.expand_dims(block_out, axis=1)
    block_list.append(block_out)
    group_list.append(group)
    se_list.append(se)

  if deform_conv:  # deform_conv
    prefix = "block_deformable_2"
    block_out = block_factory_se(input_symbol=data, name=prefix, num_filter=num_filter, group=1,
                                 stride=1,se=0, k_size=3, type='deform_conv',dim_match=dim_match)

    block_out = mx.sym.expand_dims(block_out, axis=1)
    block_list.append(block_out)
    group_list.append(prefix)
    se_list.append("se")
  speed_list= []
  for i, sym in enumerate(block_list):

    mod = mx.mod.Module(symbol=sym, context=[ctx], data_names=['data'], label_names=None)
    mod.bind(data_shapes=[['data', (1, ) + input_shape]], for_training=False)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2),
                      allow_missing=True, allow_extra=True)

    data = mx.nd.random.normal(shape=(1, ) + input_shape, ctx=ctx)
    _dataiter = mx.io.NDArrayIter(data={'data': data},
                                  batch_size=1)
    tmp_data = _dataiter.next()
    mod.forward(tmp_data)
    mod.get_outputs()[0].asnumpy()
    # tic
    start = time.time()
    for _ in range(times):
      mod.forward(tmp_data)
      y = mod.get_outputs()
      y[0].asnumpy()
    # toe
    end = time.time()
    speed = 1.0 * (end - start) / times * 1000
    speed_list.append(speed)
    msg = "Block[%d] group %s se %s  speed %f" % (i,group_list[i],se_list[i], speed)
    print(msg)
  print(' '.join([str(t) for t in speed_list]))



if __name__ == '__main__':
  # speed_test((16, 108, 108), 1, 16)
  # speed_test((16, 108, 108), 2, 24)
  # speed_test((24, 54, 54), 1, 24) # 3 layer
  # speed_test((24, 54, 54), 2, 32)
  # speed_test((32, 27, 27), 1, 32) # 3 layer
  # speed_test((32, 27, 27), 2, 64)
  # speed_test((64, 14, 14), 1, 64) # 3 layer
  # speed_test((64, 14, 14), 1, 112)
  # speed_test((112, 14, 14), 1, 112) # 3 layer
  # speed_test((112, 14, 14), 2, 184)
  # speed_test((184, 7, 7), 1, 184) # 3 layertrue
  #speed_test((184, 7, 7), 1, 352)
  
  # speed_test_se((64, 54, 54),stride=1, num_filter=256,dim_match = False,deform_conv = False) 
  # speed_test_se((256, 54, 54),stride=1, num_filter=256,dim_match = True,deform_conv = False) # 2 layer
  
  # speed_test_se((256, 54, 54),stride=2, num_filter=512,dim_match = False,deform_conv = False)
  # speed_test_se((512, 27, 27),stride=1, num_filter=512,dim_match = True,deform_conv = False) # 3 layer
  
  # speed_test_se((512, 27, 27),stride=2, num_filter=1024,dim_match = False,deform_conv = False)
  # speed_test_se((1024, 14, 14),stride=1, num_filter=1024,dim_match = True,deform_conv = False)  # 5 layer
# 
  # speed_test_se((1024, 14, 14),stride=2, num_filter=2048,dim_match = False,deform_conv = False) 
  speed_test_se((2048, 7, 7),stride=1, num_filter=2048,dim_match = True,deform_conv = True)  # 2 layer
  
  #speed_test_se((256, 54, 54),2, 512,dim_match = False,shape = True)
