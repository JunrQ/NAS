import mxnet as mx
import numpy as np

from blocks import block_factory
from blocks_se import block_factory_se

def sample_fbnet(theta_path, feature_dim=192, data=None, prefix='fbnet'):
    if len(prefix) > 0:
        prefix += '_'
    if data is None:
        data = mx.symbol.Variable(name="data")
    with open(theta_path, 'r') as f:
        lines = f.readlines()
    tbs_idx = 0

    _f = [16, 16, 24, 32, 
        64, 112, 184, 352,
        1984]
    _n = [1, 1, 4, 4,
        4, 4, 4, 1,
        1]

    _s = [1, 1, 2, 2,
        2, 1, 2, 1,
        1]
    _e = [1, 1, 3, 6,
        1, 1, 3, 6]
    _kernel = [3, 3, 3, 3,
            5, 5, 5, 5]
    _group = [1, 2, 1, 1,
            1, 2, 1, 1]
    _tbs = [1, 7]
    _block_size = len(_e) + 1
    for outer_layer_idx in range(len(_f)):
        num_filter = _f[outer_layer_idx]
        num_layers = _n[outer_layer_idx]
        s_size = _s[outer_layer_idx]

        if outer_layer_idx == 0:
            data = mx.sym.Convolution(data=data, num_filter=num_filter,
                    kernel=(3, 3), stride=(s_size, s_size), pad=(1, 1),
                    name=prefix+'conv0')
            data = mx.sym.Activation(data=data, act_type='relu', name=prefix+'relu0')
            input_channels = num_filter
        elif (outer_layer_idx <= _tbs[1]) and (outer_layer_idx >= _tbs[0]):
            for inner_layer_idx in range(num_layers):
                
                if inner_layer_idx == 0:
                    s_size = s_size
                else:
                    s_size = 1
                # tbs part
                line = lines[tbs_idx]
                theta = [float(tmp) for tmp in line.strip().split(' ')[1:]]
                block_idx = np.argmax(theta)
                
                if block_idx != _block_size - 1:
                    kernel_size = (_kernel[block_idx], _kernel[block_idx])
                    group = _group[block_idx]
                    prefix_ = "%s_layer_%d_%d_block_%d" % (prefix, outer_layer_idx, inner_layer_idx, block_idx)
                    expansion = _e[block_idx]
                    stride = (s_size, s_size)

                    data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9,
                                    name="%slayer_%d_%d_bn" % (prefix, outer_layer_idx, inner_layer_idx))

                    block_out = block_factory(data, input_channels=input_channels,
                                        num_filters=num_filter, kernel_size=kernel_size,
                                        prefix=prefix_, expansion=expansion,
                                        group=group, # shuffle=True,
                                        stride=stride, bn=False)
                    if (input_channels == num_filter) and (s_size == 1):
                        block_out = block_out + data
                    data = block_out
                tbs_idx += 1
                input_channels = num_filter
        
        elif outer_layer_idx == len(_f) - 1:
            # last 1x1 conv part
            data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9,
                                    name="%slayer_out_bn" % (prefix))
            data = mx.sym.Activation(data=data, act_type='relu', name="%sout_relu0" % prefix)
            data = mx.sym.Convolution(data, num_filter=num_filter,
                                    stride=(s_size, s_size),
                                    kernel=(3, 3),
                                    name="%slayer_%d_last_conv" % (prefix, outer_layer_idx))
        else:
            raise ValueError("Wrong layer index %d" % outer_layer_idx)
        
    # avg pool part
    data = mx.symbol.Pooling(data=data, global_pool=True, 
        kernel=(7, 7), pool_type='avg', name=prefix+"global_pool")

    data = mx.symbol.Flatten(data=data, name=prefix+'flat_pool')
    data = mx.symbol.FullyConnected(data=data, num_hidden=feature_dim,
                                    name=prefix+'flat')
    return data

def sample_fbnet_se(theta_path, feature_dim=192, data=None, prefix='fbnet'):
  if len(prefix) > 0:
    prefix += '_'
  if data is None:
    data = mx.symbol.Variable(name="data")
  with open(theta_path, 'r') as f:
    lines = f.readlines()
  
  _unistage = 4
  tbs_idx = 0

  _n = [3, 4, 6, 3]
  _f = [64, 256, 512, 1024, 2048]
  _bottle_neck = [1, 1, 0, 0, 0]
  _se =    [0, 0, 0, 1, 0]
  _kernel =[3, 3, 3, 3, 3]
  _group = [1, 2, 1, 1, 2]

  _block_size = len(_group)

  # data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, 
  #             momentum=0.9, name=prefix+'bn0')
  data = mx.sym.Convolution(data=data, num_filter=_f[0], kernel=(3, 3), stride=(1, 1), 
            pad=(3, 3),no_bias=True, name=prefix+"conv0")
  data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, 
            momentum=0.9, name=prefix+'bn1')
  data = mx.sym.Activation(data=data, act_type='relu', name=prefix+'relu0')
  data = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), 
            pool_type='max')

  for b_index in range(_unistage):

    num_layers = _n[b_index]
    num_filter = _f[b_index+ 1]

    for l_index in range(num_layers):

      if l_index == 0:
        stride =2
        dim_match = False
      else:
        stride =1
        dim_match = True

      line = lines[tbs_idx]
      theta = [float(tmp) for tmp in line.strip().split(' ')[1:]]
      i_index = np.argmax(theta)
      tbs_idx += 1

      if b_index >= 3 and l_index >= 1 and i_index == _block_size:  # deformable_Conv part
        prefix = "layer_%d_%d_block_defConv" % (b_index, l_index)
        data = block_factory_se(input_symbol=data, name=prefix, num_filter=num_filter, 
                                group=1, stride=1, se=0, k_size=3, type='deform_conv')
        
      else:
        type = 'bottle_neck' if i_index <= 1 else 'resnet'
        prefix = "layer_%s_%d_%d_block_%d" % (type, b_index, l_index, i_index)
        group = _group[i_index]
        kernel_size = _kernel[i_index]
        se = _se[i_index]

        data = block_factory_se(input_symbol=data, name=prefix, 
                        num_filter=num_filter, group=group, stride=stride,
                        se=se, k_size=kernel_size, type=type, dim_match=dim_match)

  # avg pool part
  data = mx.symbol.Pooling(data=data, global_pool=True,
                            kernel=(7, 7), pool_type='avg', name=prefix+"global_pool")

  data = mx.symbol.Flatten(data=data, name=prefix+'flat_pool')
  data = mx.symbol.FullyConnected(data=data, num_hidden=feature_dim)
