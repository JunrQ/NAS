import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import socket
import sys
import math

class Config(object):
  init_channels = 16
  layers = 18
  train_portion = 0.7
  initial_temp = 2.5
  anneal_rate = 0.99
  epochs = 100
  clip_gradient = 2.5
  lr_arch = 0.001
  lr_model = 0.01
  wd_model = 5e-4
  wd_arch = 1e-4
  resource_constraint_weight = 1e-8
  cutout = True
  save_arch_frequence = 5
  input_shape = (3, 32, 32)
  anneal_frequence = 50
  if cutout:
    cutout_length = 16

from torch.optim.lr_scheduler import _LRScheduler

class CosineDecayLR(_LRScheduler):
  def __init__(self, optimizer, T_max, alpha=1e-4,
               t_mul=2, lr_mul=0.9,
               last_epoch=-1,
               warmup_step=300,
               logger=None):
    self.T_max = T_max
    self.alpha = alpha
    self.t_mul = t_mul
    self.lr_mul = lr_mul
    self.warmup_step = warmup_step
    self.logger = logger
    self.last_restart_step = 0
    self.flag = True
    super(CosineDecayLR, self).__init__(optimizer, last_epoch)

    self.min_lrs = [b_lr * alpha for b_lr in self.base_lrs]
    self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]

  def get_lr(self):
    T_cur = self.last_epoch - self.last_restart_step
    assert T_cur >= 0
    if T_cur <= self.warmup_step and (not self.flag):
      base_lrs = [min_lr + rise_lr * T_cur
              for (base_lr, min_lr, rise_lr) in 
                zip(self.base_lrs, self.min_lrs, self.rise_lrs)]
      if T_cur == self.warmup_step:
        self.last_restart_step = self.last_epoch
        self.flag = True
    else:
      base_lrs = [self.alpha + (base_lr - self.alpha) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs]
    if T_cur == self.T_max:
      self.last_restart_step = self.last_epoch
      self.min_lrs = [b_lr * self.alpha for b_lr in self.base_lrs]
      self.base_lrs = [b_lr * self.lr_mul for b_lr in self.base_lrs]
      self.rise_lrs = [1.0 * (b - m) / self.warmup_step 
                     for (b, m) in zip(self.base_lrs, self.min_lrs)]
      self.T_max = int(self.T_max * self.t_mul)
      self.flag = False
    
    return base_lrs

class AvgrageMeter(object):

  def __init__(self, name=''):
    self.reset()
    self._name = name

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
  
  def __str__(self):
    return "%s: %.5f" % (self._name, self.avg)
  
  def __repr__(self):
    return self.__str__()

def weights_init(m, deepth=0, max_depth=2):
  if deepth > max_depth:
    return
  if isinstance(m, torch.nn.Conv2d):
    torch.nn.init.kaiming_uniform_(m.weight.data)
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.Linear):
    m.weight.data.normal_(0, 0.01)
    if m.bias is not None:
      m.bias.data.zero_()
  elif isinstance(m, torch.nn.BatchNorm2d):
    return
  elif isinstance(m, torch.nn.ReLU):
    return
  elif isinstance(m, torch.nn.Module):
    deepth += 1
    for m_ in m.modules():
      weights_init(m_, deepth)
  else:
    raise ValueError("%s is unk" % m.__class__.__name__)

def loss(model, input, target, temperature, criterion):
  """Given input, lable, temperature and criterion return loss.
  """
  logits, _, _ = model(input, temperature)
  return criterion(logits, target) 

def credit(model, input, target, temperature, criterion):
  """Credits SNAS search gradients assign to each structural decision.
  """
  loss_ = loss(model, input, target, temperature, criterion)
  dL = torch.autograd.grad(loss_, input)[0]
  dL_dX = dL.view(-1)
  X = input.view(-1)
  credit = torch.dot(dL_dX.double() , X.double())
  return credit

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
ip = get_ip()

import logging
from datetime import datetime
from termcolor import colored
class _MyFormatter(logging.Formatter):
    """Copy from tensorpack.
    """
    def format(self, record):
        date = colored('IP:%s '%str(ip), 'yellow') + colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def _getlogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger
_logger = _getlogger()
def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')
def _set_file(path):
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        _logger.info("Existing log file '{}' backuped to '{}'".
            format(path, backup_name))
    hdl = logging.FileHandler(filename=path,
        encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(hdl)
