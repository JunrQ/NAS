import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch

from data import FBNet_ds

class FaceData(data.Dataset):
  """For face training.
  """
  def __init__(self, root, lst_file,
               num_sample_classes=1000,
               save_rst='face_sampled.data'):
    """
    Parameters
    ----------
    root : str
      root path
    lst_file : str
      lst file path
    num_sample_classes : int
      number of classes sampled
    save_rst : str
      it may takes a long time to read, so save sampled dataset
    """
    pass

def get_face_ds(args, traindir,
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
  normalize = transforms.Normalize(mean=[i/255.0 for i in [123.0, 116.0, 103.0]],
                                   std=[1.0 * 255 / 100] * 3)
  
  ds_folder = FBNet_ds(root=traindir, 
          transform=transforms.Compose([
          transforms.RandomResizedCrop(108),
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