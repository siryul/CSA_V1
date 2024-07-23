import torch
import numpy as np
from torch.utils.data import Dataset
from os import path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .sampler import ClassAwareSampler


class LT_Dataset(Dataset):
  num_classes = 7

  def __init__(self, root, txt, transform=None) -> None:
    self.img_path = []
    self.targets = []
    self.transform = transform
    with open(txt) as f:
      for line in f:
        self.img_path.append(path.join(root, line.split()[0]))
        self.targets.append(int(line.split()[1]))

    cls_num_list_old = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]
    sorted_classes = np.argsort(-np.array(cls_num_list_old))
    self.class_map = [0 for i in range(self.num_classes)]
    for i in range(self.num_classes):
      self.class_map[sorted_classes[i]] = i

    self.targets = np.array(self.class_map)[self.targets].tolist()

    self.class_data = [[] for i in range(self.num_classes)]
    for i in range(len(self.targets)):
      j = self.targets[i]
      self.class_data[j].append(i)

    self.cls_num_list = [np.sum(np.array(self.targets) == i) for i in range(self.num_classes)]

  def __getitem__(self, index):
    path = self.img_path[index]
    target = self.targets[index]
    print('ham10000', path, target)

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    return sample, target

  def __len__(self):
    return len(self.targets)

  def get_cls_num_list(self):
    return self.cls_num_list


class LT_Dataset_Eval(Dataset):
  num_classes = 7

  def __init__(self, root, txt, class_map, transform=None):
    self.img_path = []
    self.targets = []
    self.transform = transform
    self.class_map = class_map
    with open(txt) as f:
      for line in f:
        self.img_path.append(path.join(root, line.split()[0]))
        self.targets.append(int(line.split()[1]))

    self.targets = np.array(self.class_map)[self.targets].tolist()

  def __len__(self):
    return len(self.targets)

  def __getitem__(self, index):
    path = self.img_path[index]
    target = self.targets[index]

    with open(path, 'rb') as f:
      sample = Image.open(f).convert('RGB')
    if self.transform is not None:
      sample = self.transform(sample)
    return sample, target


class HAM10000:

  def __init__(self, distributed, root, batch_size=60, num_works=4, randaug=False):
    normalize = transforms.Normalize(mean=[], std=[])

    transform_train = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ])

    transform_test = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])

    curr_path = path.dirname(path.abspath(__file__))

    train_txt = path.join(curr_path, "./data_txt/ham10000_train.txt")
    eval_txt = path.join(curr_path, "./data_txt/ham10000_test.txt")

    train_dataset = LT_Dataset(root, train_txt, transform=transform_train)
    eval_dataset = LT_Dataset_Eval(root,
                                   eval_txt,
                                   transform=transform_test,
                                   class_map=train_dataset.class_map)

    self.cls_num_list = train_dataset.cls_num_list

    self.dist_sampler = DistributedSampler(train_dataset) if distributed else None
    self.train_instance = DataLoader(train_dataset,
                                     batch_size,
                                     shuffle=True,
                                     num_workers=num_works,
                                     pin_memory=True,
                                     sampler=self.dist_sampler)
    self.eval_instance = DataLoader(eval_dataset,
                                    batch_size,
                                    shuffle=False,
                                    num_workers=num_works,
                                    pin_memory=True)

    balance_sampler = ClassAwareSampler(train_dataset)
    self.train_balance = DataLoader(train_dataset,
                                    batch_size,
                                    shuffle=False,
                                    num_workers=num_works,
                                    pin_memory=True,
                                    sampler=balance_sampler)

    self.eval = DataLoader(eval_dataset,
                           batch_size,
                           shuffle=False,
                           num_workers=num_works,
                           pin_memory=True)


if __name__ == '__main__':
  dataset = HAM10000(False, '/Volumes/T7/data/HAM10000')
  print(dataset.train_instance)
