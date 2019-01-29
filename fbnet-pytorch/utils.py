import socket
import logging
import sys
import os
from termcolor import colored
from datetime import datetime
import shutil

import torch

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

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
ip = get_ip()

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
