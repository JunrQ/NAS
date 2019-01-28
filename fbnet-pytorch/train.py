
import torch
from torch import nn

import logging
import argparse
import time
import os

from model import Trainer, FBNet
from data import get_ds
from blocks import get_blocks
from utils import _logger, _set_file


class Config(object):
  num_cls_used = 100
  init_theta = 1.0
  alpha = 0.2
  beta = 0.6
  speed_f = './speed.txt'
  w_lr = 0.01
  w_mom = 0.9
  w_wd = 1e-4
  t_lr = 0.001
  t_wd = 3e-3
  init_temperature = 5.0
  temperature_decay = 0.965
  model_save_path = '/mnt/data3/zcq/nas/fbnet-pytorch/'
  total_epoch = 90
  start_w_epoch = 10

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

imagenet_root = '/mnt/data1/caiyang/imagenet/train/'
train_queue, val_queue = get_ds(args, imagenet_root,
                                num_cls_used=config.num_cls_used)

blocks = get_blocks()
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
                  logger=_logger)

trainer.search(train_queue, val_queue,
               total_epoch=config.total_epoch,
               start_w_epoch=config.start_w_epoch,
               log_frequence=args.log_frequence)
