import numpy as np
import torch 
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import time
import logging
import argparse

import utils
from utils import *

def train(train_queue, valid_queue, model, criterion, optimizer_arch, 
          optimizer_model, lr_arch, lr_model, cfg, logger=None, 
          batch_num=-1, log_frequence=50):
  objs = utils.AvgrageMeter()
  policy  = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  # if logger is not None:
  #   logger.info("Start new epoch training")
  # criterion = criterion.cuda()
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = input.cuda()
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
    total_loss = policy_loss + value_loss + \
                 resource_constraint*cfg.resource_constraint_weight
    total_loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), cfg.clip_gradient)
    optimizer_arch.step()
    optimizer_model.step()

    prec1, prec5 = utils.accuracy(logit, target, topk=(1, 5))
    objs.update(value_loss.data, n)
    policy.update(policy_loss.data, n)
    top1.update(prec1.data , n)
    top5.update(prec5.data , n)

    if logger is not None:
      if step > 0 and step % log_frequence == 0:
        if batch_num > 0:
          logger.info("[Step] %d/%d [loss] %.6f [acc] %.4f" % (step, batch_num, 
                          value_loss.detach().cpu().numpy(), \
                          prec1.detach().cpu().numpy() / 100.0))
        else:
          logger.info("[Step] %d [loss] %.6f [acc] %.4f" % (step, 
                          value_loss.detach().cpu().numpy(), 
                          prec1.detach().cpu().numpy() / 100.0))

  return top1.avg, top5.avg, objs.avg, policy.avg

def infer(valid_queue, model, criterion, cfg, logger=None, batch_num=-1,
          log_frequence=10):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  # logger.info("Start new epoch inference")

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()

    temperature = cfg.initial_temp * np.exp(-cfg.anneal_rate * step)
    logits , _, _ = model(input , temperature)
    loss = criterion(logits, target)
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))

    n = input.size(0)
    objs.update(loss.data , n)
    top1.update(prec1.data , n)
    top5.update(prec5.data , n)
    if logger is not None:
      if step > 0 and step % log_frequence == 0:
        if batch_num > 0:
          logger.info("[Step] %d/%d [loss] %.6f [acc] %.4f" % (step, batch_num, 
                          loss.detach().cpu().numpy(),
                          prec1.detach().cpu().numpy() / 100.0))
        else:
          logger.info("[Step] %d [loss] %.6f [acc] %.4f" % (step, 
                          loss.detach().cpu().numpy(),
                          prec1.detach().cpu().numpy() / 100.0))
  return top1.avg, top5.avg, objs.avg
