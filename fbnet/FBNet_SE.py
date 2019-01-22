"""Implementation of FBNet.
"""
import time
import logging
import mxnet as mx
import numpy as np
from functools import reduce
import os
# from mxnet.moduel.executor_group import DataParallelExecutorGroup
from blocks_se import block_factory_se
from blocks import block_factory, block_factory_test
from util import sample_gumbel, ce, CosineDecayScheduler_Grad
import glob
from util import _logger

class FBNet_SE(object):
  def __init__(self, batch_size, output_dim,
               alpha=0.2, beta=0.6,
               feature_dim=192,
               model_type='amsoftmax',
               input_shape=(3, 108, 108),
               label_shape=None,
               data_name='data',
               label_name='softmax_label',
               logger=_logger,
               ctxs=[mx.cpu()],
               initializer=mx.initializer.Normal(),
               theta_unique_name='theta_arc',
               batch_end_callback=None,
               eval_metric=None,
               log_frequence=50,
               save_frequence=5000,
               eps=1e-10,
               num_examples=200000,
               theta_init_value=1.0,
               work_load_list=None,
               load_model_path = None,
               save_model_path = None,
               kvstore='local'):
    """
    Parameters
    ----------
    batch_size : int
      batch size for training
    output_dim : int
      output dimensions of last fc layer, number of classes actually
    alpha : float
      loss parameters, default is 0.2
    beta : float
      loss aprameters, default is 0.6
    feature_dim : int
      dimensions, default is 192
    model_type : str
      for now, support `softmax`, `amsoftmax`, `arcface`,
      softmax mean original fc
    input_shape : tuple
      input shape, CHW
    data_name : str
      name for inpu
    label_name : str
      name for label
    logger : 
    ctxs : list of mx.Context
    initializer : 
    theta_unique_name : str
      used for recognizing $\theta$ parameters
    
    """
    self._unistage = 4

    self._n = [3,4,6,3]
    self._f = [64, 256, 512, 1024, 2048]
    self._bottle_neck = [1,1,0,0,0]
    self._se =    [0, 0, 0, 1, 0]
    self._kernel =[3, 3, 3, 3, 3]
    self._group = [1, 2, 1, 1, 2]

    self._block_size = len(self._group)

    self._theta_vars = []
    self._batch_size = batch_size
    self._save_frequence = save_frequence
    self._eps = eps

    self._data = mx.sym.var(data_name)
    self._label = mx.sym.var(label_name)
    self._label_shape = label_shape
    self._output_dim = output_dim
    self._alpha = alpha
    self._beta = beta
    self._temperature = mx.sym.var("temperature")
    self._m = []
    self._m_size = []
    self._binded = False
    self._logger = logger
    if not isinstance(ctxs, list):
      ctxs = [ctxs]
    self._ctxs = ctxs
    self._init = initializer
    self._data_name = data_name
    self._label_name = label_name
    self._theta_unique_name = theta_unique_name
    self._batch_end_callback = batch_end_callback
    self._log_frequence = log_frequence
    self._theta_name = []
    self._b_name = []
    self._gumbel_vars = []
    self._gumbel_var_names = []
    self._num_examples = num_examples
    self._w_updater = None
    self._theta_updater = None
    self._feature_dim = feature_dim
    self._model_type = model_type
    self._b = dict()
    self._theta_init_value = theta_init_value
    self._save_model_path = save_model_path
    self._load_model_path = load_model_path
    # TODO(ZhouJ) use kvstore for updating
    self._kvstore = kvstore
    self._avg_grad_dict = None

    if isinstance(eval_metric, list):
      eval_metric_list = []
      for tmp in eval_metric:
        eval_metric_list.append(mx.metric.create(tmp))
      self._eval_metric = eval_metric_list
    elif isinstance(eval_metric, str):
      self._eval_metric = [mx.metric.create(eval_metric)]
    elif eval_metric is None:
      self._eval_metric = None
    else:
      raise ValueError("Unsupported eval matrix type %s" % str(type(eval_metric)))

    if isinstance(input_shape, list):
      input_shape = tuple(input_shape)
    if work_load_list is None:
      assert self._batch_size % len(self._ctxs) == 0
    else:
      raise NotImplementedError("work load list is not supported for now")
    len_ctx = len(self._ctxs)
    self._dev_batch_size = self._batch_size / len_ctx
    self._input_shapes = {data_name: (self._batch_size / len_ctx, ) + input_shape,
                          label_name: (self._batch_size / len_ctx, ) if label_shape is None else \
                                      (self._batch_size / len_ctx, ) + self._label_shape,
                          "temperature": (1, )}

    assert model_type in ['softmax', 'amsoftmax', 'arcface']
    if model_type != 'softmax':
      self._label_index = mx.sym.var("label_index")
      self._input_shapes.setdefault('label_index', (self._batch_size, ))

  def init_optimizer(self, lr_decay_step=None, cosine_decay_step=None):
    """Init optimizer, define updater.
    """
    optimizer_params_w = {'learning_rate':0.005,
                          'momentum':0.9,
                          'clip_gradient': 10.0,
                          'wd':1e-4,
                          'sym': self._loss,
                          'rescale_grad': 1.0 / self._batch_size}
    batch_num = self._num_examples / self._batch_size
    self._batch_num = batch_num
    if lr_decay_step is not None:
      steps = [int(batch_num * i) for i in lr_decay_step]
      lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
          step=steps,
          factor=0.5)
      optimizer_params_w.setdefault("lr_scheduler", lr_scheduler)
    if cosine_decay_step is not None:
      
      lr_scheduler = CosineDecayScheduler_Grad(
        first_decay_step=cosine_decay_step,
        t_mul=2.0, m_mul=0.95, alpha=0.001, base_lr=0.005,rise_region=300)
      # lr_scheduler = mx.lr_scheduler.CosineDecayScheduler(
      #     first_decay_step=cosine_decay_step,
      #     t_mul=2.0, m_mul=0.9, alpha=0.0001, base_lr=optimizer_params_w['learning_rate'])

      optimizer_params_w.setdefault("lr_scheduler", lr_scheduler)
    self._w_updater = mx.optimizer.get_updater(
      mx.optimizer.create('sgd', **optimizer_params_w))

    optimizer_params_theta={'learning_rate':0.01,
                            'wd':5e-4}
    self._theta_updater = mx.optimizer.get_updater(
      mx.optimizer.create('adam', **optimizer_params_theta))

  def _build_Se(self):
    """Build symbol.
    """
    self._logger.info("Build symbol")
    data = self._data

    data = mx.sym.Convolution(data=data, num_filter=self._f[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),no_bias=True, name="conv0")
    data = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=0.9, name= 'bn0')
    data = mx.sym.Activation(data=data, act_type='relu', name='relu0')
    data = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0', 
                                 pooling_convention='full')

    for b_index in range(self._unistage):

      num_layers = self._n[b_index]
      num_filter = self._f[b_index+ 1]

      for l_index in range(num_layers):
        tmp_name = "layer_%d_%d_%s" % (b_index, l_index,self._theta_unique_name)
        tmp_gumbel_name = "layer_%d_%d_%s" % (b_index, l_index, "gumbel_random")
        self._theta_name.append(tmp_name)

        if b_index != 0 and  l_index ==0:
          stride =2
        else:
          stride =1
        dim_match = False if l_index == 0 else True
        block_list = []
        for i_index in range(self._block_size):
          
          type = 'bottle_neck' if i_index<=1 else 'resnet'
         
          prefix = "layer_%s_%d_%d_block_%d" % (type,b_index, l_index, i_index)
          #self._logger.info('layer_params %s'%prefix)
          group = self._group[i_index]
          kernel_size = self._kernel[i_index]
          se = self._se[i_index]

          block_out = block_factory_se(input_symbol = data,name= prefix,num_filter=num_filter,group=group,stride=stride,
                                       se= se,k_size=  kernel_size,type = type,dim_match= dim_match)

          block_out = mx.sym.expand_dims(block_out, axis=1)
          block_list.append(block_out)
         # TODO  deformable_Conv part
        if b_index>=3 and l_index>=1: 
          prefix = "layer_%d_%d_block_defConv" % (b_index, l_index)
          self._logger.warn("name %s "%prefix)

          theta_var = mx.sym.var(tmp_name, shape=(self._block_size + 1,))
          gumbel_var = mx.sym.var(tmp_gumbel_name, shape=(self._block_size + 1,))
          self._input_shapes[tmp_name] = (self._block_size + 1,)
          self._input_shapes[tmp_gumbel_name] = (self._block_size + 1,)
          # TODO
          block_out = block_factory_se(input_symbol=data, name=prefix, num_filter=num_filter, group=1,
                                       stride=2,se=1, k_size=3, type='deform_conv')
          
          block_out = mx.sym.expand_dims(block_out, axis=1)
          block_list.append(block_out)

          self._m_size.append(self._block_size+1)
        else:
          theta_var = mx.sym.var(tmp_name, shape=(self._block_size,))
          gumbel_var = mx.sym.var(tmp_gumbel_name, shape=(self._block_size,))
          self._m_size.append(self._block_size )
          self._input_shapes[tmp_name] = (self._block_size ,)
          self._input_shapes[tmp_gumbel_name] = (self._block_size ,)

        self._theta_vars.append(theta_var)
        self._gumbel_vars.append(gumbel_var)
        self._gumbel_var_names.append([tmp_gumbel_name, self._m_size[-1]])

        theta = mx.sym.broadcast_div(mx.sym.elemwise_add(theta_var, gumbel_var), self._temperature)

        m = mx.sym.repeat(mx.sym.reshape(mx.sym.softmax(theta), (1, -1)),
                          repeats=self._dev_batch_size, axis=0)
        self._m.append(m)
        m = mx.sym.reshape(m, (-2, 1, 1, 1))
        # TODO why stack wrong
        data = mx.sym.concat(*block_list, dim=1, name="layer_%d_%d_concat" % (b_index, l_index))
        data = mx.sym.broadcast_mul(data, m)
        data = mx.sym.sum(data, axis=1)

    # avg pool part
    data = mx.symbol.Pooling(data=data, global_pool=True,
                             kernel=(7, 7), pool_type='avg', name="global_pool")

    data = mx.symbol.Flatten(data=data, name='flat_pool')
    data = mx.symbol.FullyConnected(data=data, num_hidden=self._feature_dim)
    # fc part
    if self._model_type == 'softmax':
      data = mx.symbol.FullyConnected(name="output_fc",
                                      data=data, num_hidden=self._output_dim)
    elif self._model_type == 'amsoftmax':
      s = 30.0
      margin = 0.3
      data = mx.symbol.L2Normalization(data, mode='instance', eps=1e-8) * s
      w = mx.sym.Variable('fc_weight', init=mx.init.Xavier(magnitude=2),
                          shape=(self._output_dim, self._feature_dim), dtype=np.float32)
      norm_w = mx.symbol.L2Normalization(w, mode='instance', eps=1e-8)
      data = mx.symbol.AmSoftmax(data, weight=norm_w, num_hidden=self._output_dim,
                                 lower_class_idx=0, upper_class_idx=self._output_dim,
                                 verbose=False, margin=margin, s=s,
                                 label=self._label_index)
    elif self._model_type == 'arcface':
      s = 64.0
      margin = 0.5
      data = mx.symbol.L2Normalization(data, mode='instance', eps=1e-8) * s
      w = mx.sym.Variable('fc_weight', init=mx.init.Xavier(magnitude=2),
                          shape=(self._output_dim, self._feature_dim), dtype=np.float32)
      norm_w = mx.symbol.L2Normalization(w, mode='instance', eps=1e-8)
      data = mx.symbol.Arcface(data, weight=norm_w, num_hidden=self._output_dim,
                               lower_class_idx=0, upper_class_idx=self._output_dim,
                               verbose=False, margin=margin, s=s,
                               label=self._label_index)
    self._output = data

  def define_loss(self):
    self._logger.info("Define loss")
    self._softmax_output = mx.sym.SoftmaxActivation(self._output)
    ce = -self._label * mx.sym.log(self._softmax_output + self._eps) - \
      (1.0 - self._label) * mx.sym.log(1.0 - self._softmax_output + self._eps)
    ce = mx.sym.sum(ce)
    with open('speed_se.txt', 'r') as f:
      speed_f = f.readlines()
    for l in range(len(self._m)):
      b_l_i = []
      speed_b_tmp = speed_f[l].strip().split(' ')
      for i in range(self._m_size[l]):
        #b_tmp = 2.0 * (0.9 ** l) * (1.1 ** i)
        b_tmp = float(speed_b_tmp[i])
        b_l_i.append(b_tmp)
      self._b["b_%d" % l] = mx.nd.array(b_l_i)
      self._input_shapes["b_%d" % l] = (self._m_size[l], )
      self._b_name.append("b_%d" % l)

      b_l = mx.sym.var("b_%d" % l, shape=(self._m_size[l], ))
      b_l = mx.sym.reshape(b_l, (1, -1))
      if l == 0:
        lat = mx.sym.sum(self._m[l] * mx.sym.repeat(b_l, 
                                repeats=self._dev_batch_size, axis=0))
      else:
        lat = lat + mx.sym.sum(self._m[l] * mx.sym.repeat(b_l, 
                                repeats=self._dev_batch_size, axis=0))
    # loss = ce * self._alpha * (mx.sym.log(lat) ** self._beta)
    loss = ce + self._alpha * (mx.sym.log(lat) ** self._beta)
    self._loss = mx.sym.make_loss(loss)
    self._loss = mx.sym.Group([self._loss, mx.sym.BlockGrad(self._softmax_output)])
    return self._loss
  
  def bind_exe(self, **input_shape):
    """Bind symbol to get an executor.
    """
    self._logger.info("Bind executor")
    self._input_names = [self._data_name] + [self._label_name] + \
                        self._b_name + [i[0] for i in self._gumbel_var_names]
    self._arg_names = self._loss.list_arguments()
    self._param_names = [x for x in self._arg_names if x not in self._input_names]
    self.grad_req = {}
    for k in self._arg_names:
      if k in self._param_names:
        self.grad_req[k] = 'write'
      else:
        self.grad_req[k] = 'null'

    self._exe = []
    for ctx in self._ctxs:
      self._exe.append(self._loss.simple_bind(ctx,
                      grad_req=self.grad_req,
                      **self._input_shapes))

    self._param_arrays = []
    self._grad_arrays = []
    self._arg_dict = []
    self._aux_dict = []
    self._grad_dict = []

    init_dict = {}
    start_epoch = 0
    for i, exe in enumerate(self._exe):

      self._param_arrays.append(exe.arg_arrays)
      self._grad_arrays.append(exe.grad_arrays)
      self._arg_dict.append(exe.arg_dict)
      self._grad_dict.append(exe.grad_dict)
      self._aux_dict.append(exe.aux_dict)

      if self._load_model_path is None:
        # initilize parameters
        # default value for $\theta$ is 1
        for name, arr in self._arg_dict[i].items():
          if name not in self._input_shapes:
            if name in init_dict:
              arr[:] = init_dict[name].as_in_context(arr.context)
            else:
              # TODO there is a warning
              self._init(name, arr)
              init_dict[name] = arr.as_in_context(mx.cpu())
          elif self._theta_unique_name in name:
            arr[:] = self._theta_init_value
      else:

        arg_params, aux_params, start_epoch = self.load_model()

        for name, arr in self._arg_dict[i].items():
          if name not in self._input_shapes:
            self._arg_dict[i][name][:] = arg_params[name]

        for name, arr in self._aux_dict[i].items():
          if name not in self._input_shapes:
            self._aux_dict[i][name][:] = aux_params[name]

        self._logger.info("Success load_model for build executor")

    self._no_update_params_name = set((self._data_name, self._label_name,
            "temperature"))
    
    for k, _ in self._b.items():
        self._no_update_params_name.add(k)
      
    for k in self._gumbel_var_names:
      if not k[0] in self._arg_dict[0].keys():
        break
      self._no_update_params_name.add(k[0])

    # Malloc for Multi-Card Gradient Memory
    if self._avg_grad_dict is None:
      self._avg_grad_dict = {}
      self._ctx_idx_grad_dict = {}

    avg_dict_size = [0] * len(self._ctxs)
    for n, p in zip(self._arg_names, self._param_arrays[0]):
      if not (n in self._no_update_params_name):
        ctx_idx = np.argmin(avg_dict_size)
        avg_dict_size[ctx_idx] += p.size
        tmp_array = mx.nd.zeros(p.shape, self._ctxs[ctx_idx])
        self._avg_grad_dict[n] = tmp_array
        self._ctx_idx_grad_dict[n] = ctx_idx

    return start_epoch

  def forward_backward(self, data, label, temperature=5.0):
    data_slice = []
    label_slice = []
    for i in range(len(self._ctxs)):
      data_slice.append(data[i*self._dev_batch_size:(i+1)*self._dev_batch_size])
      label_slice.append(label[i*self._dev_batch_size:(i+1)*self._dev_batch_size])
    
    gumbel_list = []
    for k in self._gumbel_var_names:
      if not k[0] in self._arg_dict[0].keys():
        break
      tmp_gumbel = sample_gumbel((k[1], ))
      gumbel_list.append(1.0 * mx.nd.array(tmp_gumbel))

    for i in range(len(self._ctxs)):
      self._arg_dict[i][self._data_name][:] = data_slice[i]
      if self._model_type != 'softmax':
        label_index = label_slice[i]
        self._arg_dict[i]['label_index'][:] = label_index
      label_ = mx.nd.one_hot(label_slice[i], self._label_shape[0])
      self._arg_dict[i][self._label_name][:] = label_
      if "temperature" in self._arg_dict[i].keys():
        self._arg_dict[i]["temperature"][:] = temperature

      for k, v in self._b.items():
        self._arg_dict[i][k][:] = v
      
      for idx, k in enumerate(self._gumbel_var_names):
        if not k[0] in self._arg_dict[i].keys():
          break
        self._arg_dict[i][k[0]][:] = gumbel_list[idx]
        # self._arg_dict[i][k[0]][:] = 1.0 * mx.nd.zeros((k[1]))

      self._exe[i].forward(is_train=True)
      self._exe[i].backward()

  def update_w_a(self, data, label, temperature=5.0):
    """Update parameters of $w_a$
    """
    self.forward_backward(data, label, temperature)
    len_ctx = len(self._ctxs)

    for i in range(len(self._param_arrays[0])):
      name = self._arg_names[i]
      if (not self._theta_unique_name in name) and \
        (not name in self._no_update_params_name):
        ctx_idx = self._ctx_idx_grad_dict[name]
        for idx_ in range(ctx_idx, len_ctx + ctx_idx):
          idx = idx_ % len_ctx
          weight = self._param_arrays[idx][i]
          if idx == ctx_idx:
            z = self._avg_grad_dict[name]
            for tmp in self._grad_arrays:
              z = tmp[i].copyto(z)
              self._grad_arrays[idx][i] += z
            grad_ = self._grad_arrays[idx][i]
            grad_ = grad_ / len_ctx
          grad = grad_.as_in_context(weight.context)
          self._w_updater(i * len(self._ctxs) + idx_ - ctx_idx, grad, weight)

  def update_theta(self, data, label, temperature):
    """Update $\theta$.
    """
    self.forward_backward(data, label, temperature)
    len_ctx = len(self._ctxs)

    for i in range(len(self._param_arrays[0])):
      name = self._arg_names[i]
      if self._theta_unique_name in name and \
         (not name in self._no_update_params_name):
        ctx_idx = self._ctx_idx_grad_dict[name]
        for idx_ in range(ctx_idx, len_ctx + ctx_idx):
          idx = idx_ % len_ctx
          weight = self._param_arrays[idx][i]
          if idx == ctx_idx:
            z = self._avg_grad_dict[name]
            for tmp in self._grad_arrays:
              z = tmp[i].copyto(z)
              self._grad_arrays[idx][i] += z
            grad_ = self._grad_arrays[idx][i]
            grad_ = grad_ / len_ctx

          grad = grad_.as_in_context(weight.context)
          self._w_updater(i * len(self._ctxs) + idx_ - ctx_idx, grad, weight)

  def _train(self, dataset, epochs, updater_func, start_epoch=0):
    assert isinstance(dataset, mx.io.DataIter)
    n_batches = self._log_frequence * self._batch_size
    for epoch_ in range(epochs):
      epoch = epoch_ + start_epoch
      log_tic = time.time()

      end_of_batch = False
      nbatch = 0
      data_iter = iter(dataset)
      next_data_batch = next(data_iter)

      while not end_of_batch:
        data_batch = next_data_batch
        data_input = data_batch.data[0]
        label_input = data_batch.label[0]

        updater_func(data_input, label_input)
        try:
          # pre fetch next batch
          next_data_batch = next(data_iter)
        except StopIteration:
          end_of_batch = True
        
        if nbatch > 1 and (nbatch % self._log_frequence == 0):
          log_toc = time.time()
          speed = 1.0 * n_batches /  (log_toc - log_tic)
          eval_str = ''
          total_loss = [exe.outputs[0].asnumpy().sum() for exe in self._exe]
          total_loss = reduce(lambda x, y: x + y, total_loss) / self._batch_size
          
          label_ = label_input.asnumpy()
          
          pred_ = [exe.outputs[1].asnumpy() for exe in self._exe]
          pred_ = np.concatenate(pred_, axis=0)
          # print(pred_)

          loss = ce(label_, pred_) / self._batch_size
          pred_a = np.argmax(pred_, axis=1)
          acc = 1.0 * np.sum(label_ == pred_a) / self._batch_size
          eval_str += "acc: %f" % acc
          
          # TODO
          # if self._eval_metric is not None:
          #   for eval_m in self._eval_metric:
          #     eval_m.update(label_input, self._softmax_output)
          #     eval_result = eval_m.get()
          #     eval_str += "[%s: %f]" % (eval_result[0], eval_result[1])
          
          self._logger.info("Epoch[%d] Batch[%d] Speed: %.3f samples/sec loss: %f ce: %f %s" % 
                            (epoch, nbatch, speed, total_loss, loss, eval_str))
          log_tic = time.time()
        
        if nbatch > 1 and (nbatch % self._save_frequence == 0):
          save_dict = {}
          # TODO save checkpoint
          if os.path.exists(self._save_model_path) is False:
            os.mkdir(self._save_model_path)

          # Symbol.save  json_file
          json_name = os.path.join(self._save_model_path,'checkpoint-{}-symbol.json').format(self._model_type)
          self._loss.save(json_name)

          save_dict = {('arg:%s' % k): v for k, v in self._exe[0].arg_dict.items()}
          save_dict.update({('aux:%s' % k): v for k, v in self._exe[0].aux_dict.items()})

          # tempe = mx.nd.array([self._temperature])
          # save_dict.update({'temperature:': tempe})
          # res = []
          # name = []
          # for t in self._theta_name:
          #  nd = self._arg_dict[t]
          #  res.append(nd)
          #  name.append(t)

          # save_dict.update({('theta:%s' % k): v for k, v in zip(name,res)})

          param_name = os.path.join(self._save_model_path, 'checkpoint_{}_{}_{}.params').format(self._model_type, epoch, nbatch)

          mx.nd.save(param_name, save_dict)
          self._logger.info('Save model %s in epoch[%d] batch[%d]' %(param_name,epoch, nbatch))

        if end_of_batch:
          # TODO do end of batch checkpoint or anything else
          pass
        
        if self._batch_end_callback is not None:
          # TODO
          pass
        
        nbatch += 1
      dataset.reset()

  def train_w_a(self, dataset, epochs=1, start_epoch=0, temperature=5.0):
    """Train parameters $w_a$.
    """
    if epochs <= 0:
      return
    self._logger.info("Start to train w_a for %d epochs from %d" %
                      (epochs, start_epoch))
    updater_func = lambda x, y: self.update_w_a(x, y, temperature)
    self._train(dataset, epochs=epochs, updater_func=updater_func,
                start_epoch=start_epoch)

  def train_theta(self, dataset, epochs=1, start_epoch=0, temperature=5.0):
    """Train parameters $\theta$.
    """
    if epochs <= 0:
      return
    self._logger.info("Start to train theta for %d epochs from %d" %
                      (epochs, start_epoch))
    updater_func = lambda x, y: self.update_theta(x, y, temperature)
    self._train(dataset, epochs=epochs, updater_func=updater_func,
                start_epoch=start_epoch)

  def get_theta(self, save=True, save_path='./theta.txt'):
    """Return theta as list of np.ndarray.
    """
    res = []
    name = []
    for t in self._theta_name:
      nd = self._arg_dict[0][t]
      res.append(nd.asnumpy().flatten())
      name.append(t)
    
    if save:
      save_f = open(save_path, 'w')
      for arr, nam in zip(res, name):
        s = nam + ': '
        s += ' '.join([str(float(i)) for i in arr])
        save_f.write(s + '\n')
      save_f.close()
    return res, name

  def load_model(self):
    if self._load_model_path is not None:

      if '.params' in self._load_model_path:
        load_param_dict = mx.nd.load(self._load_model_path)
        model_file = self._load_model_path
        epoch = self._load_model_path.split('_')[-2]
      else:
        model_list = glob.glob(self._load_model_path+'/*.params')
        model_list.sort()
        model_file = model_list[-1]
        epoch = model_list[-1].split('_')[-2]
        load_param_dict = mx.nd.load(model_list[-1])

      self._logger.info("load_model: %s  epoch[%s]"%(model_file,epoch))
      arg_params = {}
      aux_params = {}
      for k,v in load_param_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
          arg_params[name] = v
        elif arg_type == 'aux':
            aux_params[name] = v
        else:
          logging.info("Not found param_type %s" %k)

    return arg_params,aux_params,epoch

  def search(self, 
             w_s_ds,
             theta_ds,
             init_temperature=5.0,
             temperature_annel=0.956, # exp(-0.045)
             epochs=90,
             start_w_epochs=10,
             result_prefix='',
             lr_decay_step=None,
             cosine_decay_step=None):
    """Find optimial $\theta$
    """
    if len(result_prefix) > 0:
      result_prefix += '_'
    self._build_Se()
    self.define_loss()

    start_epoch = self.bind_exe()
    start_epoch = int(start_epoch)

    logging.info("start_epoch %d" % start_epoch)

    self.init_optimizer(lr_decay_step=lr_decay_step,
                        cosine_decay_step=cosine_decay_step)
    
    
    w_epochs = int(start_w_epochs - start_epoch -1 )
    if w_epochs > 0:
      self.train_w_a(w_s_ds, epochs = w_epochs,
                     start_epoch=start_epoch, temperature=init_temperature)
      temperature = init_temperature               
    else:
      temperature = init_temperature * pow(temperature_annel, -1 * w_epochs)
    
    start_epoch = max(start_w_epochs - 1, start_epoch)

    for epoch in range(start_epoch, epochs):
      self.train_w_a(w_s_ds, epochs=1, start_epoch=epoch,
                       temperature=temperature)
      self.train_theta(theta_ds, epochs=1, start_epoch=epoch,
                         temperature=temperature)
      self.get_theta(save_path="./theta-result/%sepoch_%d_theta.txt" %
                                 (result_prefix, epoch))
      temperature *= temperature_annel
