from layers import *
from torch import nn

from functools import reduce
import time

def tensor2list(t):
  """Transfer torch.tensor to list.
  """
  return list(t.detach().long().numpy())
def scalar2int(t):
  """Transfer scalar to int.
  """
  return int(t.detach().long().numpy())

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
    self.params = torch.tensor(torch.randn(self._num_blks), requires_grad=True)

    self.latency = [-1] * self._num_blks
  
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

    for i, p in enumerate(self.params):
      self.register_parameter('arch_param_layer_%d' % i, p)

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
      return x, g, idx


class Trainer(object):
  def __init__(self,
               first_conv_filter=32,
               num_layers=12,
               stride=[3, 6, 9],
               filters=[64, 128, 512, 1024],
               feature_dim=192,
               num_classes=100,
               w_lr=0.01,
               w_mom=0.9,
               w_wd=1e-4,
               t_lr=0.001,
               t_wd=3e-3,
               t_beta=(0.5, 0.999)):
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
      reduction = (i in [stride])
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
    
    self.ce_loss = torch.nn.CrossEntropyLoss()
    self.arch_params = self.proxyless.params

    self.w_params = []
    for n, p in self.proxyless.named_parameters():
      if not 'arch_param_layer' in n:
        self.w_params.append(p)
    
    self.w_opt = torch.optim.SGD(
                    self.w_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)
    
    self.t_opt = torch.optim.Adam(
                    self.arch_params,
                    lr=t_lr, betas=t_beta,
                    weight_decay=t_wd)
    


  
  def train_arch(self, input, target):
    """Train architecture parameters.

    """
    pass
  
  def _step_train(self, input, target):
    """Perform one step of $w$ training.
    """
    output = self.proxyless(input, search=False, train=True)

    loss = self.ce_loss(output, target)

    loss.backward()


  def _step_search(self, input, target):
    """Perform one step of $\alpha$ training.
    """
    output, g, idx = self.proxyless(input, search=True, train=False)

    loss = self.ce_loss(output, target)

    loss.backward()

    for tbs_idx, g_ind in enumerate(g):
      grad_ind = g_ind.grad

      alpha = self.params[tbs_idx]

      alpha.backward(grad_ind)












