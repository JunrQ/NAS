

class AvgrageMeter(object):

  def __init__(self, name=''):
    self.reset()
    self._name = name

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
  
  def __str__(self):
    return "%s: .6%f" % (self._name, self.avg)
  
  def __repr__(self):
    return self.__str__()