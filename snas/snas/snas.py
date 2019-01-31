
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import DataParallel
import math
import time
import logging

from ops import *
from utils import AvgrageMeter, weights_init, CosineDecayLR

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

  def forward(self, x, Z):
    res = [op(x) for op in self._ops]
    output = sum(z * tmp for z, tmp in zip(Z, res))
    cost = sum(z * c for z, c in zip(Z, self.COSTS))
    return output, cost


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
    cost_ = sum(costs[-self._multiplier:]) # / self._multiplier
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
    steps : int
      block steps
    multiplier : int
      multiplier
    stem_multiplier : int
      multiplier of first conv num_filter
    """
    super(SNAS,self).__init__()

    self._num_classes = num_classes
    self._steps = steps
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
    return logits, costs / input.size()[0]
  
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

  def arch_parameters(self):
    return self._arch_parameters

  def model_parameters(self):
    params = self.named_parameters()
    res = []
    for k in params:
      if not ('alphas_nomal' in k[0] and 'alphas_reduce' in k[0]):
        res.append(k[1])
    return res

class Trainer(object):
  """Training network parameters and alpha.
  """
  def __init__(self, network,
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
               save_theta_prefix='',
               resource_weight=0.001):
    assert isinstance(network, SNAS)
    network.apply(weights_init)
    network = network.train().cuda()
    self._criterion = nn.CrossEntropyLoss().cuda()

    alpha_params = network.arch_parameters()
    mod_params = network.model_parameters()
    self.alpha = alpha_params
    if isinstance(gpus, str):
      gpus = [int(i) for i in gpus.strip().split(',')]
    network = DataParallel(network, gpus)
    self._mod = network
    self.gpus = gpus

    self.w = mod_params
    self._tem_decay = temperature_decay
    self.temp = init_temperature
    self.logger = logger
    self.save_theta_prefix = save_theta_prefix
    self._resource_weight = resource_weight

    self._loss_avg = AvgrageMeter('loss')
    self._acc_avg = AvgrageMeter('acc')
    self._res_cons_avg = AvgrageMeter('resource-constraint')

    self.w_opt = torch.optim.SGD(
                    mod_params,
                    w_lr,
                    momentum=w_mom,
                    weight_decay=w_wd)
    self.w_sche = CosineDecayLR(self.w_opt, **lr_scheduler)
    self.t_opt = torch.optim.Adam(
                    alpha_params,
                    lr=t_lr, betas=t_beta,
                    weight_decay=t_wd)

  def _acc(self, logits, target):
    batch_size = target.size()[0]
    pred = torch.argmax(logits, dim=1)
    acc = torch.sum(pred == target).float() / batch_size
    return acc

  def train(self, input, target):
    """Update parameters.
    """
    self.w_opt.zero_grad()
    logits, costs = self._mod(input, self.temp)
    acc = self._acc(logits, target)
    costs = costs.mean()
    costs *= self._resource_weight
    loss = self._criterion(logits, target) + costs
    loss.backward()
    self.w_opt.step()
    self.t_opt.step()
    return loss, costs, acc
  
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
    loss, res_cost, acc = func(input, target)

    # Get status
    batch_size = input.size()[0]

    self._loss_avg.update(loss)
    self._res_cons_avg.update(res_cost)
    self._acc_avg.update(acc)

    if step > 1 and (step % log_frequence == 0):
      self.toc = time.time()
      speed = 1.0 * (batch_size * log_frequence) / (self.toc - self.tic)

      self.logger.info("Epoch[%d] Batch[%d] Speed: %.6f samples/sec %s %s %s" 
              % (epoch, step, speed, self._loss_avg, self._acc_avg,
                 self._res_cons_avg))
      map(lambda avg: avg.reset(), [self._loss_avg, self._res_cons_avg,
                                    self._acc_avg])
      self.tic = time.time()
  
  def search(self, train_ds,
            epochs=90,
            log_frequence=100):
    """Search model.
    """

    self.tic = time.time()
    for epoch in range(epochs):
      self.logger.info("Start to train for epoch %d" % (epoch))
      for step, (input, target) in enumerate(train_ds):
        self._step(input, target, epoch, 
                   step, log_frequence,
                   lambda x, y: self.train(x, y))
      self.save_alpha('./alpha-result/%s_theta_epoch_%d.txt' % 
                  (self.save_theta_prefix, epoch))
      self.decay_temperature()
      self.w_sche.step()

  def save_alpha(self, save_path='alpha.txt'):
    """Save alpha.
    """
    res = []
    with open(save_path, 'w') as f:
      for i, t in enumerate(self.alpha):
        n = 'normal' if i == 0 else 'reduce'
        assert i <= 1
        tmp = t.size(0)
        f.write(n + ':' + '\n')
        for j in range(tmp):
          t_list = list(t[j].detach().cpu().numpy())
          res.append(t_list)
          s = ' '.join([str(tmp) for tmp in t_list])
          f.write(s + '\n')
    return res