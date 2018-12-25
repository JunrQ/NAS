import shutil
from datetime import datetime
from termcolor import colored
import time
import numpy as np
import socket
import logging
import sys
import os
import mxnet as mx
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    assert x.ndim == 2
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
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

def get_train_ds(args, kv=None):
    if kv is None:
        rank = 0
        nworker = 1
    else:
        rank, nworker = kv.rank, kv.num_workers
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.train_rec_path,
        label_width         = 1,
        mean_r              = 123.0,
        mean_g              = 116.0,
        mean_b              = 103.0,
        scale               = 0.01,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = tuple(map(int, args.image_shape.split(','))),
        batch_size          = args.batch_size,
        resize_height       = 400,
        resize_width        = 400,
        patch_size          = args.patch_size,
        patch_idx           = args.patch_idx,
        do_aug              = True,
        aug_seq             = 'aug_face',
        FacePatchSize_Main  = 267,
        FacePatchSize_Other = 128,
        PatchFullSize        = 128 if args.image_shape.split(',')[-1] == '108' else 256,
        PatchCropSize        = 108 if args.image_shape.split(',')[-1] == '108' else 224,
        illum_trans_prob    = args.illum_trans_prob,
        gauss_blur_prob     = 0.3,
        motion_blur_prob    = 0.1,
        jpeg_comp_prob    = 0.4,
        res_change_prob    = 0.4,
        hsv_adjust_prob    = args.hsv_adjust_prob,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank,
        force2gray = args.force2gray,
        force2color = args.force2color,
        isgray='true' if args.isgray else 'false')
    return train

def inv_gumbel_cdf(y, mu=0.0, beta=1.0, eps=1e-20):
    y = np.array(y)
    return mu - beta * np.log(-np.log(y + eps))

def sample_gumbel(shape):
    p = np.random.random(shape)
    return inv_gumbel_cdf(p)


import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
import gzip, struct
import errno
import subprocess

def download_file(url, local_fname=None, force_write=False):
    # requests is not default installed
    import requests
    if local_fname is None:
        local_fname = url.split('/')[-1]
    if not force_write and os.path.exists(local_fname):
        return local_fname

    dir_name = os.path.dirname(local_fname)

    if dir_name != "":
        if not os.path.exists(dir_name):
            try: # try to create the directory if it doesn't exists
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_fname

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

def ce(label, pred):
    label = label.ravel()
    assert label.shape[0] == pred.shape[0]
    prob = pred[np.arange(label.shape[0]), np.int64(label)]
    return -np.log(prob + 1e-9).sum()

# Cosine Decay scheduler
# class CosineDecayScheduler(LRScheduler):
#     """ Cosine Decay Scheduler.
#     (ZhouJ)

#     Parameters
#     ----------
#        base_lr: float, base learning rate
#        first_decay_step: int, decay steps, T_0
#        t_mul: float
#        m_mul: float
#        alpha: float, lr_min / lr_max

#     """
#     def __init__(self, first_decay_step, t_mul=2.0, m_mul=0.5, alpha=0.001, base_lr=0.01):
#         super(CosineDecayScheduler, self).__init__(base_lr)
#         assert isinstance(first_decay_step, int)
#         if first_decay_step < 1:
#             raise ValueError("decay_step must be strictly positive")
#         self.global_step = 0
#         self.T_0 = first_decay_step
#         self.t_mul = t_mul
#         self.m_mul = m_mul
#         self.init_lr = base_lr
#         self.alpha = alpha

#     def __call__(self, num_update):
#         T_cur = num_update - self.global_step
#         assert T_cur >= 0
#         if T_cur >= self.T_0:
#             # restart
#             self.global_step = num_update
#             self.init_lr *= self.m_mul
#             self.base_lr = self.init_lr
#             self.T_0 = int(self.T_0 * self.t_mul)
#         else:
#             cosine_decay = 0.5 * (1 + math.cos(math.pi * T_cur / self.T_0))
#             decayed = (1 - self.alpha) * cosine_decay + self.alpha
#             self.base_lr = self.init_lr * decayed
#         return self.base_lr