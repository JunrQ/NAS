
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import time
import logging

from ops import *
from utils import AvgrageMeter, weights_init, CosineDecayLR
from data_parallel import DataParallel

class MixedOp(nn.Module):
  def __init__(self, C, stride, h, w, ratio=1.0):
    """Mixed Operator.

    Parameters
    ----------
    C : int
      input channels = num_filters = C
    stride : int
    h : int
      height
    w : int 
      width
    ratio : flot
      cost = flop + max * ratio
    """
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.COSTS = []
    self.height = h
    self.width = w
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'sep' in primitive:
        FLOP = 0
        MAC = 0
        FLOP1 = (self.height * self.width * op.op[1].kernel_size[0]**2 \
                 * op.op[1].in_channels * op.op[1].out_channels )
        FLOP1 = FLOP / op.op[1].groups
        FLOP2 = (self.height * self.width * op.op[1].kernel_size[0]**2)
        FLOP = (FLOP1 + FLOP2) * 2
        MAC1 = self.height * self.width * (op.op[1].in_channels + op.op[1].out_channels)
        MAC1 = MAC1 + (op.op[1].in_channels * op.op[1].out_channels) 
        MAC2 = MAC1 + (op.op[1].in_channels * op.op[1].out_channels)/ op.op[1].groups
        MAC = (MAC1 + MAC2) * 2

      if 'dil' in primitive:
        FLOP = (self.width * self.height * op.op[1].in_channels * op.op[1].out_channels\
                * op.op[1].kernel_size[0] ** 2) / op.op[1].groups
        FLOP += (self.width * self.height * op.op[2].in_channels * op.op[2].out_channels) 
        
        MAC1 = (self.width * self.height) * (op.op[1].in_channels + op.op[1].in_channels)
        MAC1 += (op.op[1].kernel_size[0] ** 2 *op.op[1].in_channels * op.op[1].out_channels) / op.op[1].groups
        MAC2 = (self.width * self.height) * (op.op[2].in_channels + op.op[2].in_channels)
        MAC2 += (op.op[2].kernel_size[0] ** 2 *op.op[2].in_channels * op.op[2].out_channels) / op.op[2].groups
        
        MAC = MAC1 + MAC2
        # op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      if 'skip' in primitive:
        FLOP = 0
        MAC = 2 * C * self.height * self.width 
      if 'none' in primitive:
        FLOP = 0
        MAC = 0
      self._ops.append(op)
      self.COSTS.append(FLOP + ratio * MAC)

    self.cost = torch.tensor(sum(self.COSTS) / len(PRIMITIVES), 
                             requires_grad=False).cuda()

  def forward(self, x, Z):
    output = sum(z * op(x) for z, op in zip(Z, self._ops))
    return output, self.cost


class Cell(nn.Module):
  def __init__(self, steps, multiplier, C_prev_prev, 
               C_prev, C, reduction, reduction_prev, 
               h, w):
    """Cell.

    Parameters
    ----------
    steps : int

    """
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2 + i):
        stride = 2 if reduction and j < 2 else 1
        op  = MixedOp(C, stride, h, w)
        self._ops.append(op)

  def forward(self, s0, s1, Z):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    costs = []
    offset = 0
    for _ in range(self._steps):
      output_ = [self._ops[offset+j](h, Z[offset+j]) for j, h in enumerate(states)]
      s = sum([tmp[0] for tmp in output_])
      cost = sum([tmp[1] for tmp in output_])
      offset += len(states)
      states.append(s)
      costs.append(cost)
    state = torch.cat(states[-self._multiplier:], dim=1)
    cost_ = torch.mean(torch.tensor(costs[-self._multiplier:]))
    return state, cost_

class SNAS(nn.Module):
  def __init__(self, C, num_classes, layers, 
               steps=4, multiplier=4, stem_multiplier=3,
               input_channels=3, shape=(3, 108, 108)):
    """
    Parameters
    ----------
    C : int
      num_filters of first conv
    num_classes : int
      number of classes of output
    layers : int
      number of layers
    criterion
    steps : int
      block steps
    multiplier : int
      multiplier
    stem_multiplier : int
      multiplier of first conv num_filter
    """
    super(SNAS,self).__init__()

    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._steps = steps
    self._multiplier = multiplier
    h, w = shape[1], shape[2] # Just for calculating flop, mac
    assert input_channels == shape[0]

    C_curr = stem_multiplier * C 
    self.stem = nn.Sequential(
      nn.Conv2d(input_channels, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr))

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    # Build arch
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
        h, w = [int(math.ceil(1.0 * x / 2)) for x in [h, w]]
        assert h >= 7 and w >= 7
      else:
        reduction = False 
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, 
                  reduction, reduction_prev, h, w)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
      
  def forward(self, input, temperature):
    s0 = s1 = self.stem(input)
    costs = 0
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        Z = self.architect_dist(self.alphas_reduce , 
                                  temperature)
      else:
        Z = self.architect_dist(self.alphas_normal , 
                                  temperature)
      s0, (s1, cost) = s1, cell(s0, s1, Z)
      costs += cost
    out = self.global_pooling(s1) 
    logits = self.classifier(out.view(out.size(0), -1))
    return logits, costs
  
  def architect_dist(self, alpha, temperature):
    """Given temperature return a relaxed one hot
    """
    weight = nn.functional.gumbel_softmax(alpha,
                                temperature)
    return weight

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = torch.nn.Parameter(1e-3 * torch.randn(k, num_ops).cuda(), 
        requires_grad=True)
    self.alphas_reduce = torch.nn.Parameter(1e-3 * torch.randn(k, num_ops).cuda(), 
        requires_grad=True)
    self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
    ]

  @property
  def arch_parameters(self):
    return self._arch_parameters

  def model_parameters(self):
    # Thanks to Chuanhong Huang
    params = self.named_parameters()
    return [p for n, p in params if n not in ['alphas_nomal', 'alphas_reduce']]

class Trainer(object):
  """Training network parameters and theta separately.
  """
  def __init__(self, network,
               criterion,
               w_lr=0.01,
               w_mom=0.9,
               w_wd=1e-4,
               t_lr=0.001,
               t_wd=3e-3,
               t_beta=(0.5, 0.999),
               init_temperature=5.0,
               temperature_decay=0.965,
               logger=logging,
               lr_scheduler={'T_max' : 200},
               gpus=[0],
               save_theta_prefix=''):
    assert isinstance(network, SNAS)
    network.apply(weights_init)
    network = network.train().cuda()
    if isinstance(gpus, str):
      gpus = [int(i) for i in gpus.strip().split(',')]
      network = DataParallel(network, gpus)
    self.gpus = gpus
    self._mod = network
    theta_params = network.arch_parameters
    mod_params = network.parameters()
    self.theta = theta_params
    self.w = mod_params
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    self.save_theta_prefix = save_theta_prefix
    self._criterion = criterion

    self._loss_avg = AvgrageMeter('loss')

    self.w_opt = torch.optim.SGD(
                    mod_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)
    
    self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)

    self.t_opt = torch.optim.Adam(
                    theta_params,
                    lr=t_lr, betas=t_beta,
                    weight_decay=t_wd)

  def train_w(self, input, target, decay_temperature=False):
    """Update model parameters.
    """
    self.w_opt.zero_grad()
    logits, costs = self._mod(input, self.temp)
    loss = self._criterion(logits, target) + costs

    loss.backward()
    self.w_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss
  
  def train_t(self, input, target, decay_temperature=False):
    """Update theta.
    """
    self.t_opt.zero_grad()
    logits, costs = self._mod(input, self.temp)
    loss = self._criterion(logits, target) + costs

    loss.backward()
    self.t_opt.step()
    if decay_temperature:
      tmp = self.temp
      self.temp *= self._tem_decay
      self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
    return loss
  
  def decay_temperature(self, decay_ratio=None):
    tmp = self.temp
    if decay_ratio is None:
      self.temp *= self._tem_decay
    else:
      self.temp *= decay_ratio
    self.logger.info("Change temperature from %.5f to %.5f" % (tmp, self.temp))
  
  def _step(self, input, target, 
            epoch, step,
            log_frequence,
            func):
    """Perform one step of training.
    """
    input = input.cuda()
    target = target.cuda()
    loss = func(input, target)

    # Get status
    batch_size = self._mod.batch_size

    self._loss_avg.update(loss)

    if step > 1 and (step % log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)

      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s" 
              % (epoch, step, speed, self._loss_avg))
      map(lambda avg: avg.reset(), [self._loss_avg])
      self.tic = time.time()
  
  def search(self, train_w_ds,
            train_t_ds,
            total_epoch=90,
            start_w_epoch=10,
            log_frequence=100):
    """Search model.
    """
    assert start_w_epoch >= 1, "Start to train w"
    self.tic = time.time()
    for epoch in range(start_w_epoch):
      self.logger.info("Start to train w for epoch %d" % epoch)
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()
        # print(self.w_sche.last_epoch, self.w_opt.param_groups[0]['lr'])

    self.tic = time.time()
    for epoch in range(total_epoch):
      self.logger.info("Start to train theta for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_t_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_t(x, y, False))
        self.save_theta('./theta-result/%s_theta_epoch_%d.txt' % 
                    (self.save_theta_prefix, epoch+start_w_epoch))
      self.decay_temperature()
      self.logger.info("Start to train w for epoch %d" % (epoch+start_w_epoch))
      for step, (input, target) in enumerate(train_w_ds):
        self._step(input, target, epoch + start_w_epoch, 
                   step, log_frequence,
                   lambda x, y: self.train_w(x, y, False))
        self.w_sche.step()

  def save_theta(self, save_path='theta.txt'):
    """Save theta.
    """
    res = []
    with open(save_path, 'w') as f:
      for t in self.theta:
        t_list = list(t)
        res.append(t_list)
        s = ' '.join([str(tmp) for tmp in t_list])
        f.write(s + '/n')
    return res