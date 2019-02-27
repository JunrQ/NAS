from layers import *
from torch import nn

from functools import reduce

def tensor2list(t):
  """Transfer torch.tensor to list.
  """
  return list(t.long().numpy())
def scalar2int(t):
  """Transfer scalar to int.
  """
  return int(t.long().numpy())

class MixedBlocks(nn.Module):
  """Mixed blks.
  """
  def __init__(self, reduction, in_channels, out_channels):
    """
    Parameters
    ----------
    reduction : bool
      True if stride == 2, False if stride == 1
    in_channels : int
      number of input channels
    out_channels : int
      number of output channels
    """
    # TODO there should be an edge between every two op
    assert False
    if not reduction:
      dp_conv_name = DepthConvLayer.__name__
      cfgs = [
                {'name' : dp_conv_name,
                 'kernel_size' : 3,
                 'dilation' : 2},
                {'name' : dp_conv_name,
                 'kernel_size' : 3},
                {'name' : dp_conv_name,
                 'kernel_size' : 5},
                {'name' : dp_conv_name,
                 'kernel_size' : 7},
                {'name' : IdentityLayer.__name__}
             ]
      for c in cfgs:
        c['in_channels'] = in_channels
        c['out_channels'] = out_channels
      self.blks = nn.ModuleList(
        [set_layer_from_config(cfg) for cfg in cfgs])
    else:
      dp_conv_name = DepthConvLayer.__name__
      cfgs = [
                {'name' : dp_conv_name,
                 'kernel_size' : 3,
                 'dilation' : 2},
                {'name' : dp_conv_name,
                 'kernel_size' : 3},
                {'name' : dp_conv_name,
                 'kernel_size' : 5},
                {'name' : dp_conv_name,
                 'kernel_size' : 7},
                {'name' : PoolingLayer.__name__,
                 'pool_type' : 'avg',
                 'kernel_size' : 3},
                {'name' : PoolingLayer.__name__,
                 'pool_type' : 'max',
                 'kernel_size' : 3}
             ]
      for c in cfgs:
        c['in_channels'] = in_channels
        c['out_channels'] = out_channels
        c['stride'] = 2
      self.blks = nn.ModuleList(
        [set_layer_from_config(cfg) for cfg in cfgs])
    self._num_blks = len(cfgs)

    # Define architecture parameters
    self.params = torch.tensor(torch.randn(self._num_blks), requires_grad=False)
  
  def forward(self, x, train=True, search=False):
    if train:
      assert not search
      m = torch.distributions.categorical.Categorical(self.params)
      choosen_idxs = m.sample(1)
      choosen_idxs = scalar2int(choosen_idxs)
      return self.blks[choosen_idxs](x)
    else:
      m = torch.distributions.multinomial.Multinomial(2, self.params)
      choosen_idxs = m.sample()
      while torch.any(choosen_idxs == 2):
        choosen_idxs = m.sample()

      choosen_idxs = tensor2list(choosen_idxs)
      weight = torch.softmax(self.params[choosen_idxs], 0)
      # Assume this is weighted sum
      g = [self.blks[idx] * w for idx, w in zip(choosen_idxs, weight)]
      return reduce(lambda x, y: x + y, g)


class MixedMBBlocks(nn.Module):
  """Mixed blks.
  """

  def __init__(self, reduction, in_channels, out_channels):
    """
    Parameters
    ----------
    reduction : bool
      True if stride == 2, False if stride == 1
    in_channels : int
      number of input channels
    out_channels : int
      number of output channels
    """
    
    cfgs = [
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 3,
                'expand_ratio' : 3},
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 5,
                'expand_ratio' : 3},
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 7,
                'expand_ratio' : 3},
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 3,
                'expand_ratio' : 6},
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 5,
                'expand_ratio' : 6},
              {'name' : MBInvertedConvLayer.__name__,
                'kernel_size' : 7,
                'expand_ratio' : 6},
              {'name' : IdentityLayer.__name__}
            ]
    for c in cfgs:
      c['in_channels'] = in_channels
      c['out_channels'] = out_channels
    if not reduction:
      for c in cfgs:
        c['stride'] = 1
      self.blks = nn.ModuleList(
        [MBInvertedConvLayer.build_from_config(cfg) for cfg in cfgs])
    else:
      cfgs.pop()
      for c in cfgs:  
        c['stride'] = 2
      self.blks = nn.ModuleList(
        [MBInvertedConvLayer.build_from_config(cfg) for cfg in cfgs])
    self._num_blks = len(cfgs)

    # Define architecture parameters
    self.params = torch.tensor(torch.randn(self._num_blks), requires_grad=False)
  
  def forward(self, x, train=True, search=False):
    if train:
      assert not search
      m = torch.distributions.categorical.Categorical(self.params)
      choosen_idxs = m.sample(1)
      choosen_idxs = scalar2int(choosen_idxs)
      return self.blks[choosen_idxs](x), choosen_idxs
    else:
      m = torch.distributions.multinomial.Multinomial(2, self.params)
      choosen_idxs = m.sample()
      while torch.any(choosen_idxs == 2):
        choosen_idxs = m.sample()

      choosen_idxs = tensor2list(choosen_idxs)
      weight = torch.softmax(self.params[choosen_idxs], 0)
      # Assume this is weighted sum
      g = [self.blks[idx] * w for idx, w in zip(choosen_idxs, weight)]
      return reduce(lambda x, y: x + y, g), choosen_idxs


class ProxylessNASSearcher(BasicUnit):
  """Search model arch.
  """
  def __init__(self, first_conv,
               blocks, feature_mix_layer,
               classifier):
    self.first_conv = first_conv
    self.blocks = nn.ModuleList(blocks)
    self.feature_mix_layer = feature_mix_layer
    self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = classifier
    self._num_layers = len(self.blocks)

    self.params = [b.params for b in blocks]

  def forward(self, x, train=True, search=False):
    x = self.first_conv(x)

    if train:
      assert not search
      for blk in self.blocks:
        x, idx = blk(x, train=train, search=search)
      
    else:
      g = []
      idx = []
      for blk in self.blocks:
        x, i = blk(x, train=train, search=search)
        g.append(x)
        idx.append(i)

    if self.feature_mix_layer:
			x = self.feature_mix_layer(x)
    x = self.global_avg_pooling(x)
    # x = x.view(x.size(0), -1)  # flatten

    if train:
      return x
    else:
      return x, p, idx


class Trainer(object):
  def __init__(self,
               first_conv_filter=32,
               num_layers=12,
               stride=[3, 6, 9],
               filters=[64, 128, 512, 1024],
               feature_dim=192,
               num_classes=100):
    """
    Parameters
    ----------
    first_conv_filter : int
      number of filters of first conv layer
    num_layers : int

    stride : list of int

    filters : list of int

    """
    # Build blocks
    assert len(filters) == len(stride) + 1
    self.blks = []
    in_channels = out_channels = filters[0]

    stage_count = 0
    for i in range(num_layers):
      reduction = i in [stride]
      if reduction:
        stage_count += 1
        out_channels = filters[stage_count]
      self.blks.append(MixedMBBlocks(reduction, 
              in_channels, out_channels))

      in_channels = out_channels
    
    first_conv_layer = ConvLayer(first_conv_filter, filters[0], 3)
    feature_mix_layer = ConvLayer(filters[-1], feature_dim, 3)
    classifier = LinearLayer(feature_dim, num_classes,
                              bias=False, use_bn=True,
                              ops_order='bn_weight_act',
                              act_func=None)
    self.proxyless = ProxylessNASSearcher(first_conv=first_conv_layer,
        blocks=self.blks, feature_mix_layer=feature_mix_layer,
        classifier=classifier)
  
  def train_arch(self, input, target):
    """Train architecture parameters.

    """
    pass


  def _step_search(self, input, target):

    """One step.
    """
    output, g, idx = self.proxyless(input, search=True, train=False)

    loss = 

    loss.backward()

    grad_a = g.grad










