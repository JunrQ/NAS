import numpy as np
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import os
import time
import logging
import argparse

from snas import SNAS, Trainer
from utils import _set_file, _logger
import utils
from data_face import get_face_ds

class Config(object):
  num_cls_used = 4000
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
  model_save_path = '/mnt/data3/zcq/nas/snas/face/'
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
parser.add_argument('--log-frequence', type=int, default=100,
                    help='log frequence, default is 100')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--load-model-path', type=str, default=None,
                    help='re_train, default is None')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of subprocesses used to fetch data, default is 4')

args = parser.parse_args()
config = Config()

args.model_save_path = '%s/%s/' % \
            (config.model_save_path, time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')


image_root = '/mnt/data4/zcq/face/recognition/training/imgs/MsCelebV1-Faces-Cropped/'
train_queue, val_queue, num_classes = get_face_ds(args, image_root,
                                num_cls_used=config.num_cls_used)

model = SNAS(C=config.init_channels,
             num_classes=num_classes,
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
