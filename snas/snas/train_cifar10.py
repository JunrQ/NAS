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

from snas import Network
from utils import Config as config, _set_file, _logger, \
    loss as soft_loss, credit, weights_init
import utils
from train_ops import train, infer

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of training epochs.')
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

cfg = config()
train_transform, valid_transform = utils._data_transforms_cifar10(cfg)
train_data = dset.CIFAR10(root='../../', train=True, 
                download=False, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(cfg.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=True, pin_memory=True, num_workers=16)

valid_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  pin_memory=True, num_workers=8)

