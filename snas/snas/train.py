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
parser.add_argument('--log-frequence', type=int, default=400,
                    help='log frequence, default is 400')
parser.add_argument('--patch-idx', type=int, default=0,
                    help='patch index, default is 0')
parser.add_argument('--patch-size', type=int, default=1,
                    help='patch size, default is 1')
parser.add_argument('--gpus', type=str, default='0',
                    help='gpus, default is 0')
parser.add_argument('--load-model-path', type=str, default=None,
                    help='re_train, default is None')
parser.set_defaults(
  num_classes = 81968,
  num_examples = int(3551853 * 0.8),
  image_shape='3,108,108',
  feature_dim=192,
  save_checkpoint_frequence=5000,
  restore=False, # TODO
  optimizer='sgd',
  # train_rec_path='/mnt/data4/zcq/10w/zhuzhou/MsCeleb_SrvA2_clean/MsCeleb_clean_train.rec',
  lr_decay_step=[15, 35, 60, 95],
  cosine_decay_step=3000,
)
args = parser.parse_args()
# args.model_save_path = '/mnt/data3/zcq/nas/snas/%s/' % \
#                 (time.strftime('%Y-%m-%d', time.localtime(time.time())))
args.model_save_path = '/home1/nas/snas/%s/' % \
                (time.strftime('%Y-%m-%d', time.localtime(time.time())))

if not os.path.exists(args.model_save_path):
  _logger.warn("{} not exists, create it".format(args.model_save_path))
  os.makedirs(args.model_save_path)
_set_file(args.model_save_path + 'log.log')

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
# if len(ctxs) > 1:
model = torch.nn.DataParallel(model, device_ids=ctxs)

train_transform, valid_transform = utils._data_transforms_cifar10(cfg)
train_data = dset.CIFAR10(root='../', train=True, 
                download=True, transform=train_transform)

num_train = len(train_data)
indices = list(range(num_train))
split = int(np.floor(cfg.train_portion * num_train))

train_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:5000]),
  pin_memory=True, num_workers=4)

valid_queue = torch.utils.data.DataLoader(
  train_data, batch_size=args.batch_size,
  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[10000:15000]),
  pin_memory=True, num_workers=4)

f = open("loss.txt", "w")

def train(train_queue, valid_queue, model, criterion, optimizer_arch, 
          optimizer_model, lr_arch, lr_model):
  objs = utils.AvgrageMeter()
  policy  = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  # criterion = criterion.cuda()
  for step, (input, target) in tqdm.tqdm(enumerate(train_queue)):
    model.train()
    n = input.size(0)

    # input = Variable(input, requires_grad=True).cuda()
    input = input.cuda()#.to(torch.device("cuda:1"))
    # print(input)
    # target = Variable(target, requires_grad=False).cuda(async=True)
    target = target.cuda()
    
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search , requires_grad=True).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)
    
    temperature = cfg.initial_temp * np.exp(-cfg.anneal_rate * step)

    optimizer_arch.zero_grad()
    optimizer_model.zero_grad()
    logit, _, cost = model(input , temperature)
    _, score_function, _ = model(input_search , temperature)
    
    policy_loss = torch.sum(score_function * credit(model, input_search, 
                                target_search, temperature, criterion).float())
    value_loss = criterion(logit , target)
    resource_constraint = torch.sum(cost)
    total_loss = policy_loss + value_loss + resource_constraint*cfg.resource_constraint_weight
    total_loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), cfg.clip_gradient)
    optimizer_arch.step()
    optimizer_model.step()

    prec1, prec5 = utils.accuracy(logit, target, topk=(1, 5))
    objs.update(value_loss.data, n)
    policy.update(policy_loss.data, n)
    top1.update(prec1.data , n)
    top5.update(prec5.data , n)
    return top1.avg, top5.avg, objs.avg, policy.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in tqdm.tqdm(enumerate(valid_queue)):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    temperature = cfg.initial_temp * np.exp(-cfg.anneal_rate * step)
    logits , _, cost = model(input , temperature)
    loss = criterion(logits, target)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data , n)
    top1.update(prec1.data , n)
    top5.update(prec5.data , n)
  return top1.avg, top5.avg ,objs.avg

for epoch in range(args.epochs):
  # training
  train_acc_top1, train_acc_top5 ,train_valoss, train_poloss = train(train_queue, 
        valid_queue, model, criterion, optimizer_arch, 
        optimizer_model, cfg.lr_arch, cfg.lr_model)

  # validation
  valid_acc_top1, valid_acc_top5, valid_valoss = infer(valid_queue, model, criterion)

  f.write("%5.5f  " % train_acc_top1)
  f.write("%5.5f  " % train_acc_top5)
  f.write("%5.5f  " % train_valoss)
  f.write("%5.5f  " % train_poloss) 
  f.write("%5.5f  " % valid_acc_top1) 
  f.write("%5.5f  " % valid_acc_top5) 
  f.write("%5.5f  " % valid_valoss) 
  f.write("\n")

  if epoch % 5 ==0:
    np.save("alpha_normal_" + str(epoch) + ".npy", model.alphas_normal.detach().cpu().numpy())
    np.save("alpha_reduce_" + str(epoch) + ".npy", model.alphas_reduce.detach().cpu().numpy())

  print("epoch : ", epoch ,"Train_Acc : ", train_acc_top1, "Train_value_loss : ", 
        train_valoss, "Train_policy : ", train_poloss )
  print('\n')
  print("epoch : ", epoch , "Val_Acc : ", valid_acc_top1, "Val_value_loss : ", valid_valoss)
  torch.save(model.state_dict(), 'weights.pt')
f.close()
