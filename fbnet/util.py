import shutil
from datetime import datetime
from termcolor import colored
import time
import numpy as np
import socket
import logging
import sys
import os
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