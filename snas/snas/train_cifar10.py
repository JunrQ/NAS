import numpy as np
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import os
import matplotlib.pyplot as plt
import tqdm
import time
import logging
import argparse

from snas import SNAS, Trainer
from utils import _set_file, _logger
import utils

class Config(object):
  init_channels = 16
  stacked_cell = 8
  train_portion = 0.8
  epochs = 150
  init_temperature = 2.5
  temperature_decay = 0.97
  w_lr = 0.025
  w_mom = 0.9
  w_wd = 3e-5
  t_lr = 3e-4
  t_wd = 1e-3
  t_beta = (0.5, 0.999)
  resource_constraint_weight = 1e-6
  cutout = True
  if cutout:
    cutout_length = 16

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
  'lr_mul' : 0.98,
}

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--queue-size', type=int, default=20,
                    help='train data queue size, used for shuffle.')
parser.add_argument('--model-type', type=str, default='softmax',
                    help='top model type, default is softmax')
parser.add_argument('--log-frequence', type=int, default=100,
                    help='log frequence, default is 100')
parser.add_argument('--patch-idx', type=int, default=0,
                    help='patch index, default is 0')
parser.add_argument('--patch-size', type=int, default=1,
                    help='patch size, default is 1')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--load-model-path', type=str, default=None,
                    help='re_train, default is None')

args = parser.parse_args()
args.model_save_path = '/home1/nas/snas/%s/' % \
                (time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')

config = Config()
train_transform, valid_transform = utils._data_transforms_cifar10(config)
train_data = dset.CIFAR10(root='../../', train=True, 
                download=False, transform=train_transform)

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=True, pin_memory=True, num_workers=16)

model = SNAS(C=config.init_channels,
             num_classes=10,
             layers=config.stacked_cell)

trainer = Trainer(network=model,
                  w_lr=config.w_lr,
                  w_mom=config.w_mom,
                  w_wd=config.w_wd,
                  t_lr=config.t_lr,
                  t_wd=config.t_wd,
                  t_beta=config.t_beta,
                  init_temperature=config.init_temperature,
                  temperature_decay=config.temperature_decay,
                  logger=_logger,
                  lr_scheduler=lr_scheduler_params,
                  gpus=args.gpus,
                  resource_weight=config.resource_constraint_weight)

trainer.search(train_queue,
               epochs=config.epochs,
               log_frequence=args.log_frequence)
