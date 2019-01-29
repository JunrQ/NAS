
import torch
from torch import nn
import torchvision.datasets as dset

import numpy as np
import logging
import argparse
import time
import os

from model import Trainer, FBNet
from data import get_ds
from blocks import get_blocks
from utils import _logger, _set_file


class Config(object):
  num_cls_used = 10
  init_theta = 1.0
  alpha = 0.2
  beta = 0.6
  speed_f = './speed.txt'
  w_lr = 0.1
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.01
  t_wd = 5e-4
  init_temperature = 5.0
  temperature_decay = 0.956
  model_save_path = '/home1/nas/fbnet-pytorch/'
  total_epoch = 90
  start_w_epoch = 2
  train_portion = 0.8

lr_scheduler_params = {
  'logger' : _logger,
  'T_max' : 400,
  'alpha' : 1e-4,
  'warmup_step' : 100,
  't_mul' : 1.5,
  'lr_mul' : 0.98,
}

config = Config()

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs.')
parser.add_argument('--log-frequence', type=int, default=400,
                    help='log frequence, default is 400')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--load-model-path', type=str, default=None,
                    help='re_train, default is None')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of subprocesses used to fetch data, default is 4')
args = parser.parse_args()

args.model_save_path = '%s/%s/' % \
            (config.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')

import torchvision.transforms as transforms
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
train_data = dset.CIFAR10(root='../snas/', train=True, 
                download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(config.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=True, pin_memory=True, num_workers=16)

val_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  pin_memory=True, num_workers=8)

blocks = get_blocks(cifar10=True)
model = FBNet(num_classes=config.num_cls_used,
              blocks=blocks,
              init_theta=config.init_theta,
              alpha=config.alpha,
              beta=config.beta,
              speed_f=config.speed_f)

trainer = Trainer(network=model,
                  w_lr=config.w_lr,
                  w_mom=config.w_mom,
                  w_wd=config.w_wd,
                  t_lr=config.t_lr,
                  t_wd=config.t_wd,
                  init_temperature=config.init_temperature,
                  temperature_decay=config.temperature_decay,
                  logger=_logger,
                  lr_scheduler=lr_scheduler_params,
                  gpus=args.gpus)

trainer.search(train_queue, val_queue,
               total_epoch=config.total_epoch,
               start_w_epoch=config.start_w_epoch,
               log_frequence=args.log_frequence)
