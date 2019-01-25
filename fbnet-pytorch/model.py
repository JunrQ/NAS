import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):

  def __init__(self, blocks):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for op in blocks:
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Network(nn.Module):

  def __init__(self, num_classes, criterion, blocks,
               feature_dim=192, init_theta=1.0,
               speed_f='./speed.txt',
               alpha=0.2,
               beta=0.6):
    super(Network, self).__init__()

    if isinstance(init_theta, int):
      init_func = lambda x: nn.init.constant_(x, init_theta)
    else:
      init_func = init_theta
    self._alpha = alpha
    self._beta = beta


    self._theta = []
    self._blocks = blocks
    for b in blocks:
      if len(b) > 1:
        num_block = len(b)
        theta = torch.ones((num_block, ), requires_grad=True)
        init_func(theta)
        self._theta.append(theta)

        self._ops.append(MixedOp(b))
    
    assert len(self._theta) == 22
    with open(speed_f, 'r') as f:
      self._speed = f.readlines()

    self.classifier = nn.Linear(1984, num_classes)

  def forward(self, input, temperature=5.0):
    batch_size = input.size()[0]
    data = self._blocks[0](input)
    theta_idx = 0
    lat = []
    for l_idx in range(1, len(self._blocks)):
      block = self._blocks[l_idx]
      if len(block) > 1:
        theta = self._theta[theta_idx]
        theta_idx += 1
        # t = theta.reshape(1, -1)
        t = theta.repeat(batch_size, 1)
        weight = nn.functional.gumbel_softmax(t,
                                temperature)
        speed = self._speed[theta_idx].strip().split(' ')
        speed = [float(tmp) for tmp in speed]
        lat_ = weight * torch.tensor(speed).repeat(batch_size, 1).sum()
        lat.append(lat_)

        data = self._ops[theta_idx](data, weight)

    lat = torch.tensor(lat)
    data = nn.avg_pool2d(data, data.size()[2:])
    logits = self.classifier(data)

    loss = cri() +  self._alpha * torch.sum(lat).pow(self._beta)



    return logits

