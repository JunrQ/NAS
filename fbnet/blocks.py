"""Define blocks.
"""

import mxnet as mx

def channel_shuffle(data, groups):
  data = mx.sym.reshape(data, shape=(0, -4, groups, -1, -2))
  data = mx.sym.swapaxes(data, 1, 2)
  data = mx.sym.reshape(data, shape=(0, -3, -2))
  return data

def block_factory_test(input, input_channels, 
                  num_filters,
                  prefix, kernel_size, expansion,
                  group, stride, bn=False, 
                  shuffle=True,
                  **kwargs):
  # 1*1 conv
  output = mx.sym.Convolution(data=input, num_filter=num_filters, 
                            kernel=(1, 1), stride=(1, 1), pad=(0, 0), 
                            num_group=group, no_bias=True,
                            name=prefix + '_sep_0')
  return output

def block_factory(input, input_channels, 
                  num_filters,
                  prefix, kernel_size, expansion,
                  group, stride, bn=False, 
                  shuffle=True,
                  **kwargs):
  """Return block symbol.

  Parameters
  ----------
  input : symbol
    input symbol
  input_channels : int
    number of channels of input symbol
  num_filters : int
    output channels
  prefix : str
    prefix string
  kernel_size : tuple
  expansion : int
  group : int
    conv group
  stride : tuple

  """
  # 1*1 group conv
  data = mx.sym.Convolution(data=input, num_filter=input_channels*expansion, 
                            kernel=(1, 1), stride=(1, 1), pad=(0, 0), 
                            num_group=group, no_bias=True,
                            name=prefix + '_sep_0')
  if bn:
    data = mx.sym.BatchNorm(data=data)
  data = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu0')
  if shuffle and group >= 2:
    data = channel_shuffle(data, group)
  
  # dw conv
  data = mx.sym.Convolution(data=data, num_filter=input_channels*expansion, 
                            kernel=(3, 3), stride=stride, pad=(1, 1), 
                            num_group=input_channels*expansion, no_bias=False,
                            name=prefix + '_dw')
  if bn:
    data = mx.sym.BatchNorm(data=data)
  data = mx.sym.Activation(data=data, act_type='relu', name=prefix + '_relu1')

  if shuffle:
    data = channel_shuffle(data, group)

  # 1*1 conv
  data = mx.sym.Convolution(data=data, num_filter=num_filters, 
                            kernel=(1, 1), stride=(1, 1), pad=(0, 0), 
                            num_group=group, no_bias=True,
                            name=prefix + '_sep_1')
  if bn:
    data = mx.sym.BatchNorm(data=data)
  if shuffle and group >= 2:
    data = channel_shuffle(data, group)
  
  # TODO an error occur
  # dimension match
  if (stride[0] == stride[1]  == 1) and (input_channels == num_filters):
    output = input + data
  else:
    output = data
  return data