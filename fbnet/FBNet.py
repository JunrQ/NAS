"""Implementation of FBNet.
"""
import time
import logging
import mxnet as mx
import numpy as np

from blocks import block_factory, block_factory_test
from util import sample_gumbel

class FBNet(object):
  def __init__(self, batch_size, output_dim,
               alpha=0.2, beta=0.6,
               input_shape=(3, 108, 108),
               label_shape=None,
               data_name='data',
               label_name='softmax_label',
               logger=logging,
               ctxs=mx.cpu(),
               initializer=mx.initializer.Normal(),
               theta_unique_name='theta_arc',
               batch_end_callback=None,
               eval_metric=None,
               log_frequence=50,
               save_frequence=2000,
               eps=1e-5,
               num_examples=200000):
    """
    Parameters
    ----------

    """
    self._f = [16, 16, 24, 32, 
               64, 112, 184, 352,
               1984]
    self._n = [1, 1, 4, 4,
               4, 4, 4, 1,
               1]
    self._s = [2, 1, 2, 2,
               2, 1, 2, 1,
               1]
    assert len(self._f) == len(self._n) == len(self._s)
    self._e = [1, 1, 3, 6,
               1, 1, 3, 6]
    self._kernel = [3, 3, 3, 3,
                    5, 5, 5, 5]
    self._group = [1, 2, 1, 1,
                   1, 2, 1, 1]
    assert len(self._e) == len(self._kernel) == len(self._group)
    self._block_size = len(self._e) + 1 # skip
    self._tbs = [1, 7] # include layer 7
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

    if isinstance(eval_metric, list):
      eval_metric_list = []
      for tmp in eval_metric:
        eval_metric_list.append(mx.metric.create(eval_metric))
      self._eval_metric = eval_metric_list
    elif isinstance(eval_metric, str):
      self._eval_metric = [mx.metric.create(eval_metric)]
    elif eval_metric is None:
      self._eval_metric = None
    else:
      raise ValueError("Unsupported eval matrix type %s" % str(type(eval_metric)))

    if isinstance(input_shape, list):
      input_shape = tuple(input_shape)
    self._input_shapes = {data_name: (self._batch_size, ) + input_shape,
                          label_name: (self._batch_size, ) if label_shape is None else \
                                      (self._batch_size, ) + self._label_shape,
                          "temperature": (1, )}
    

  def init_optimizer(self, optimizer='sgd', init_lr=0.01,
                     optimizer_params={'learning_rate': 0.01,},
                     lr_factor=0.1, lr_decay_step=None, **kwargs):
    """Init optimizer, define updater.
    """
    self._logger.info("Define updater with init_lr: %f " % init_lr +
                       "lr_decay_step: %s" % str(lr_decay_step))
    optimizer_params.setdefault("learning_rate", init_lr)
    if lr_decay_step is not None:
      batch_num = self._num_examples / self._batch_size
      steps = [int(batch_num * i) for i in lr_decay_step]
      lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
          step=steps,
          factor=lr_factor)
      optimizer_params.setdefault("lr_scheduler", lr_scheduler)
    updater = mx.optimizer.get_updater(
      mx.optimizer.create(optimizer, **optimizer_params))
    self._updater = updater
  
  def _build(self):
    """Build symbol.
    """
    self._logger.info("Build symbol")
    data = self._data
    for i in range(len(self._f)):
      layer_idx = i
      num_filter = self._f[i]
      num_layers = self._n[i]
      s_size = self._s[i]

      if i == 0:
        # assert self._input_shapes[1] == self._input_shapes[2] == 108
        data = mx.sym.Convolution(data=data, num_filter=self._f[i],
                  kernel=(3, 3), stride=(s_size, s_size))
        input_channels = self._f[i]
      elif i <= self._tbs[1] and i >= self._tbs[0]:
        for inner_layer_idx in range(num_layers):
          if inner_layer_idx == 0:
            s_size = s_size
          else:
            s_size = 1
          # tbs part
          block_list= []

          for block_idx in range(self._block_size - 1):
            kernel_size = (self._kernel[block_idx], self._kernel[block_idx])
            group = self._group[block_idx]
            prefix = "layer_%d_%d_block_%d" % (i, inner_layer_idx, block_idx)
            expansion = self._e[block_idx]
            stride = (s_size, s_size)

            tmp = block_factory(input=data, input_channels=input_channels,
                                num_filters=num_filter, kernel_size=kernel_size,
                                prefix=prefix, expansion=expansion,
                                group=group, shuffle=True,
                                stride=stride)
            block_list.append(tmp)
          tmp_name = "layer_%d_%d_%s" % (i, inner_layer_idx, 
                                       self._theta_unique_name)
          tmp_gumbel_name = "layer_%d_%d_%s" % (i, inner_layer_idx, "gumbel_random")
          self._theta_name.append(tmp_name)
          if inner_layer_idx >= 1: # skip part
            theta_var = mx.sym.var(tmp_name, shape=(self._block_size, ), 
                                   init=mx.init.One())
            gumbel_var = mx.sym.var(tmp_gumbel_name, shape=(self._block_size, ))
            self._input_shapes[tmp_name] = (self._block_size, )
            self._input_shapes[tmp_gumbel_name] = (self._block_size, )
            block_list.append(data)
            self._m_size.append(self._block_size)
          else:
            theta_var = mx.sym.var(tmp_name, shape=(self._block_size - 1, ),
                                   init=mx.init.One())
            gumbel_var = mx.sym.var(tmp_gumbel_name, shape=(self._block_size - 1, ))
            self._m_size.append(self._block_size - 1)
            self._input_shapes[tmp_name] = (self._block_size - 1, )
            self._input_shapes[tmp_gumbel_name] = (self._block_size - 1, )
          self._theta_vars.append(theta_var)
          self._gumbel_vars.append(gumbel_var)
          self._gumbel_var_names.append([tmp_gumbel_name, self._m_size[-1]])
          
          theta = mx.sym.broadcast_div((theta_var + gumbel_var), self._temperature)
          m = mx.sym.softmax(theta)
          
          m = mx.sym.repeat(mx.sym.reshape(m, (1, -1)), 
                            repeats=self._batch_size, axis=0)
          self._m.append(m)
          m = mx.sym.reshape(m, (-2, 1, 1, 1))
          data = mx.sym.stack(*block_list, axis=1)
          data = mx.sym.broadcast_mul(data, m)
          data = mx.sym.sum(data, axis=1)
          input_channels = num_filter
      
      elif i == len(self._f) - 1:
        # last 1x1 conv part
        data = mx.sym.Convolution(data, num_filter=num_filter,
                                  stride=(s_size, s_size),
                                  kernel=(1, 1),
                                  name="layer_%d_conv1x1" % i)
      else:
        raise ValueError("Wrong layer index %d" % i)
    
    # avg pool part
    data = mx.symbol.Pooling(data=data, global_pool=True, 
        kernel=(7, 7), pool_type='avg', name="global_pool")
  
    data = mx.symbol.Flatten(data=data, name='flat_pool')
    # fc part
    data = mx.symbol.FullyConnected(name="output_fc", 
        data=data, num_hidden=self._output_dim)
    self._output = data
  
  def define_loss(self):
    self._logger.info("Define loss")
    self._softmax_output = mx.sym.softmax(self._output)
    # TODO There is an error in softmax_cross_entropy
    # ce = mx.sym.softmax_cross_entropy(self._output,
    #                      self._label,
    #                      name='softmax_output')
    ce = -self._label * mx.sym.log(self._softmax_output + self._eps) - \
      (1 - self._label) * mx.sym.log(1 - self._softmax_output + self._eps)
    
    # TODO(ZhouJ) test time in real environment
    self._b = {}
    # lat_list = []
    for l in range(len(self._m)):
      b_l_i = []
      for i in range(self._m_size[l]):
        # b_l_i.append(2.0 * (0.8 ** l) * (1.2 ** i))
        b_l_i.append(1.0)
      self._b["b_%d" % l] = mx.nd.array(b_l_i)
      self._input_shapes["b_%d" % l] = (self._m_size[l], )
      self._b_name.append("b_%d" % l)

      b_l = mx.sym.var("b_%d" % l, shape=(self._m_size[l], ))
      b_l = mx.sym.reshape(b_l, (1, -1))
      if l == 0:
        lat = mx.sym.sum(self._m[l] * mx.sym.repeat(b_l, 
                                repeats=self._batch_size, axis=0))
      else:
        lat = lat + mx.sym.sum(self._m[l] * mx.sym.repeat(b_l, 
                                repeats=self._batch_size, axis=0))
    loss = mx.sym.sum(ce) * self._alpha * (mx.sym.log(lat) ** self._beta)
    self._loss = mx.sym.make_loss(loss)
    return self._loss
  
  def bind_exe(self, **input_shape):
    """Bind symbol to get an executor.
    """
    self._logger.info("Bind executor")
    self._input_names = [self._data_name] + [self._label_name] + \
                        self._b_name
    arg_names = self._loss.list_arguments()
    self._param_names = [x for x in arg_names if x not in self._input_names]
    self.grad_req = {}
    for k in arg_names:
      if k in self._param_names:
        self.grad_req[k] = 'write'
      elif k == self._data_name:
        self.grad_req[k] = 'null'
      else:
        self.grad_req[k] = 'null'

    # TODO Dataparaller training
    self._exe = self._loss.simple_bind(ctx=self._ctxs,
                      grad_req=self.grad_req,
                      **self._input_shapes)
    self._arg_arrays = self._exe.arg_arrays
    self._grad_arrays = self._exe.grad_arrays
    self._arg_dict = self._exe.arg_dict
    self._output_dict = self._exe.output_dict

    for name, arr in self._arg_dict.items():
      if name not in self._input_shapes:
        # TODO there is a warning
        self._init(name, arr)

  def forward_backward(self, data, label, temperature=5.0):
    self._arg_dict[self._data_name][:] = data
    if self._label_shape is not None:
      label = mx.nd.one_hot(label, self._label_shape[0])
    self._arg_dict[self._label_name][:] = label
    self._arg_dict["temperature"][:] = temperature

    self._no_update_params_name = set((self._data_name, self._label_name,
          "temperature"))
    
    for k, v in self._b.items():
      self._arg_dict[k][:] = v
      self._no_update_params_name.add(k)
    
    for k in self._gumbel_var_names:
      tmp_gumbel = sample_gumbel((k[1], ))
      self._arg_dict[k[0]][:] = 1.0 * mx.nd.array(tmp_gumbel)

    self._exe.forward(is_train=True)
    self._exe.backward()

  def update_w_a(self, data, label, temperature=5.0):
    """Update parameters of $w_a$
    """
    self.forward_backward(data, label, temperature)

    grad_dict = self._exe.grad_dict
    for i, pair in enumerate(self._arg_dict.items()):
      name, weight = pair
      if (not self._theta_unique_name in name) and \
         (not name in self._no_update_params_name):
        grad = grad_dict[name]
        self._updater(i, grad, weight)


  def update_theta(self, data, label, temperature):
    """Update $\theta$.
    """
    self.forward_backward(data, label, temperature)

    for i, pair in enumerate(zip(self._arg_dict.items(), self.grad_arrays)):
      name, weight, grad = pair
      if self._theta_unique_name in name and \
         (not name in self._b.keys()):
        self._updater(i, grad, weight)

  def _train(self, dataset, epochs, updater_func, start_epoch=0):
    assert isinstance(dataset, mx.io.DataIter)
    n_batches = self._log_frequence * self._batch_size
    for epoch_ in range(epochs):
      epoch = epoch_ + start_epoch
      epoch_tic = time.time()
      log_tic = time.time()

      end_of_batch = False
      nbatch = 0
      data_iter = iter(dataset)
      next_data_batch = next(data_iter)
      # data_name = next_data_batch.data[0][0]
      # assert len(next_data_batch.data) == 1, "Not support more than \
      #     one input, but got %d inputs" % len(next_data_batch.data)
      # assert data_name == self._data_name, "Dataset data name %s is \
      #     different from model data name %s" % (data_name, self._data_name)
      # label_name = next_data_batch.label[0][0]
      # assert len(next_data_batch.label) == 1, "Not support more than \
      #     one label, but got %d labels" % len(next_data_batch.label)
      # assert label_name == self._label_name, "Dataset label name %s is \
      #     different from model label name %s" % (label_name, self._label_name)
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

          loss = self._exe.outputs[0].asnumpy()
          eval_str = ''
          # TODO
          # if self._eval_metric is not None:
          #   for eval_m in self._eval_metric:
          #     eval_m.update(label_input, self._softmax_output)
          #     eval_result = eval_m.get()
          #     eval_str += "[%s: %f]" % (eval_result[0], eval_result[1])
          
          self._logger.info("[Epoch] %d [Batch] %d Speed: %.3f samples/batch Loss: %f %s" % 
                            (epoch, nbatch, speed, loss, eval_str))
          log_tic = time.time()
        
        if nbatch > 1 and (nbatch % self._save_frequence == 0):
          # TODO save checkpoint
          pass

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
    self._logger.info("Start to train w_a for %d epochs from %d" %
                      (epochs, start_epoch))
    updater_func = lambda x, y: self.update_w_a(x, y, temperature)
    self._train(dataset, epochs=epochs, updater_func=updater_func,
                start_epoch=start_epoch)

  def train_theta(self, dataset, epochs=1, start_epoch=0, temperature=5.0):
    """Train parameters $\theta$.
    """
    self._logger.info("Start to train theta for %d epochs from %d" %
                      (epochs, start_epoch))
    updater_func = lambda x, y: self.update_theta(x, y, temperature)
    self._train(dataset, epochs=epochs, updater_func=updater_func,
                start_epoch=start_epoch)

  def get_theta(self, save=True, save_path='./theta.txt'):
    """Return theta as list of np.ndarray.
    """
    res = []
    for t in self._theta_vars:
      res.append(t.asnumpy().reshape((-1, )))
    
    if save:
      c = 0
      save_f = open(save_path, 'w')
    
      for l in range(len(self._m_size)):
        for b in range(self._m_size[l]):
          s = "Layer: l Block: b "

          s += ' '.join(list(res[c]))
          c += 1

          save_f.write(s + '\n')
      save_f.close()
    return res

  def search(self, 
             w_s_ds,
             theta_ds,
             init_temperature=5.0,
             temperature_annel=0.956, # exp(-0.045)
             epochs=90,
             start_w_epochs=10,
             **kwargs):
    """Find optimial $\theta$
    """
    self._build()
    self.define_loss()
    self.bind_exe()
    self.init_optimizer(**kwargs)    

    self.train_w_a(w_s_ds, start_w_epochs-1,
                    start_epoch=0, temperature=init_temperature)
    temperature = init_temperature
    for epoch in range(start_w_epochs, epochs):
      self.train_w_a(w_s_ds, epochs=1, start_epoch=epoch)
      self.train_theta(theta_ds, epochs=1, start_epoch=epoch)

      self.get_theta(save_path="./epoch_%d_theta.txt" % epoch)

      temperature *= temperature_annel
