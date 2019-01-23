import numpy as np
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import tqdm
import time
import logging
import argparse

from snas import Network
from utils import Config as config, _set_file, _logger, \
    loss as soft_loss, credit
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
# args.model_save_path = '/mnt/data3/zcq/nas/snas/%s/' % \
#                 (time.strftime('%Y-%m-%d', time.localtime(time.time())))
args.model_save_path = '/home1/nas/snas/%s/' % \
                (time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')

#-----------------------------
# Model configuration
#-----------------------------
ctxs = [int(i) for i in args.gpus.strip().split(",")]
cfg = config()
CIFAR_CLASSES = 10
criterion = nn.CrossEntropyLoss().cuda()
model = Network(cfg.init_channels, CIFAR_CLASSES, cfg.layers, criterion)

optimizer_model = torch.optim.SGD(model.model_parameters(), 
    lr=cfg.lr_model, momentum=0.9, weight_decay=cfg.wd_model)
optimizer_arch = torch.optim.Adam(model.arch_parameters(), 
    lr=cfg.lr_arch, betas=(0.5, 0.999), weight_decay=cfg.wd_arch)

model.cuda()
model = torch.nn.DataParallel(model, device_ids=ctxs)

#-----------------------------
# Data set configuration
#-----------------------------
train_transform, valid_transform = utils._data_transforms_cifar10(cfg)
train_data = dset.CIFAR10(root='../', train=True, 
                download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(cfg.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  shuffle=True, pin_memory=True, num_workers=16)

valid_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  pin_memory=True, num_workers=8)

#-----------------------------
# Training and evaluation
#-----------------------------
f = open("loss.txt", "w")
for epoch in range(args.epochs):
  # training
  train_acc_top1, train_acc_top5 ,train_valoss, train_poloss = train(train_queue, 
        valid_queue, model, criterion, optimizer_arch, 
        optimizer_model, cfg.lr_arch, cfg.lr_model, cfg,
        _logger, int(num_train * cfg.train_portion / args.batch_size), args.log_frequence)

  # validation
  valid_acc_top1, valid_acc_top5, valid_valoss = infer(valid_queue, 
        model, criterion, cfg, _logger, 
        int(num_train * (1 - cfg.train_portion) / args.batch_size), args.log_frequence)

  f.write("%5.5f  " % train_acc_top1)
  f.write("%5.5f  " % train_acc_top5)
  f.write("%5.5f  " % train_valoss)
  f.write("%5.5f  " % train_poloss) 
  f.write("%5.5f  " % valid_acc_top1) 
  f.write("%5.5f  " % valid_acc_top5) 
  f.write("%5.5f" % valid_valoss) 
  f.write("\n")

  if epoch % cfg.save_arch_frequence ==0:
    np.save("alpha_normal_" + str(epoch) + ".npy", model.alphas_normal.detach().cpu().numpy())
    np.save("alpha_reduce_" + str(epoch) + ".npy", model.alphas_reduce.detach().cpu().numpy())

  msg = "[Epoch] %d [train acc] %.7f [val acc] %.7f" % (epoch, train_acc_top1, valid_acc_top1)
  torch.save(model.state_dict(), 'weights.pt')
f.close()
