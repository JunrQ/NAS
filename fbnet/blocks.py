"""Define blocks.
"""

import mxnet as mx

def block_factory(input, num_filters, prefix,
                  kernel_size, expansion,
                  group, stride, **kwargs):
  """Return block symbol.
  """
  