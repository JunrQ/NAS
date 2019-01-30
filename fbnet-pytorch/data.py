import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

class FBNet_ds(datasets.ImageFolder):
  """Image net ds folder for fbnet.
  """
  def __init__(self,
               **kwargs):
    super(FBNet_ds, self).__init__(**kwargs)
  
  def filter(self,
             samples_classes=100,
             random_seed=None):
    """Get \a samples_classes from total ds.
    """
    _num_classes =  len(self.classes)
    if not random_seed is None:
      assert isinstance(random_seed, int)
      np.random.seed(random_seed)
    choosen_cls_idx = list(np.random.choice(list(range(_num_classes)), samples_classes))
    _class_to_idx = {}
    cls_id = 0
    _cls_map = dict()
    for k, v in self.class_to_idx.items():
      if v in choosen_cls_idx:
        _class_to_idx[k] = cls_id
        _cls_map[v] = cls_id
        cls_id += 1
    self.class_to_idx = _class_to_idx

    _samples = []
    for item in self.samples:
      if item[1] in choosen_cls_idx:
        _samples.append((item[0], _cls_map[item[1]]))
    self.samples = _samples
    _classes = []
    for c in self.classes:
      if _class_to_idx.has_key(c):
        _classes.append(c)
    self.classes = _classes

def get_ds(args, traindir,
           train_portion=0.8,
           random_seed=123,
           num_cls_used=100):
  """Get data set.

  Parameters
  ----------
  args
  traindir : str
    root file dir
  train_portion : float
    train portion of total dataset
  random_seed : int
    for reproduce
  """
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  
  ds_folder = FBNet_ds(root=traindir, 
          transform=transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,]))
  if num_cls_used > 0:
    ds_folder.filter(num_cls_used, random_seed=random_seed)
    num_class = num_cls_used
  else:
    num_class = len(ds_folder.classes)

  num_train = len(ds_folder)
  indices = list(range(num_train))
  if random_seed is not None:
    np.random.seed(random_seed)
  np.random.shuffle(indices)
  split = int(np.floor(train_portion * num_train))
  
  train_queue = torch.utils.data.DataLoader(
      ds_folder, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=args.num_workers)

  valid_queue = torch.utils.data.DataLoader(
      ds_folder, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=args.num_workers)
  
  return train_queue, valid_queue, num_class
