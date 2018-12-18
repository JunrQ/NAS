import os
import argparse
import sys
import logging
from time import gmtime, strftime
sys.path.insert(0, '/home/zhouchangqing/mxnet/incubator-mxnet_9_17/python')
import mxnet as mx
from FBNet import FBNet
from util import _logger, get_train_ds, _set_file
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Train a model with data parallel for base net \
                                and model parallel for classify net.")
parser.add_argument('--batch-size', type=int, default=256,
                    help='training batch size of all devices.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs.')
parser.add_argument('--queue-size', type=int, default=20,
                    help='train data queue size, used for shuffle.')
parser.add_argument('--model-type', type=str, default='amsoftmax',
                    help='top model type, default is amsoftmax')
parser.add_argument('--log-frequence', type=int, default=400,
                    help='log frequence, default is 400')
parser.add_argument('--patch-idx', type=int, default=0,
                    help='patch index, default is 0')
parser.add_argument('--patch-size', type=int, default=1,
                    help='patch size, default is 1')
parser.set_defaults(
  num_classes=20000,
  num_examples=1284730,
  image_shape='3,108,108',
  feature_dim=192,
  conv_workspace=1024,  # this is the default value
  save_checkpoint_frequence=30000,
  restore=False,
  optimizer='sgd',
  data_nthreads=16,
  force2gray='false',
  force2color='false',
  illum_trans_prob=0.3,
  hsv_adjust_prob=0.1,
  train_rec_path='/home1/data/face_recognition/MsCeleb_train_clean1_2w.rec',
  isgray=False,
)
args = parser.parse_args()
train_w_ds = get_train_ds(args)

# TODO
# args.num_classes = 
# args.num_examples = 
# args.train_rec_path = 
train_theta_ds = get_train_ds(args)

fbnet = FBNet(batch_size=args.batch_size,
              output_dim=args.num_classes,
              label_shape=(args.num_classes, ),
              logger=_logger,
              input_shape=[int(i) for i in args.image_shape.split(',')],
              ctxs=mx.gpu(0),
              # eval_metric=['acc', 'ce'] # TODO
              num_examples=args.num_examples,
              log_frequence=args.log_frequence,
              save_frequence=args.save_checkpoint_frequence,
              feature_dim=args.feature_dim,
              model_type=args.model_type)

fbnet.search(train_w_ds, train_theta_ds)
