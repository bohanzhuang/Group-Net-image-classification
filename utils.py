import numpy as np
import os
import glob
import shutil
from random import shuffle
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import shutil


def data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])

    valid_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def unpack_data(config):

    train_file = config['train_file']
    test_file = config['test_file']
    train_data = unpickle(train_file)
    test_data = unpickle(test_file)

    return train_data, test_data


def adjust_learning_rate(args, epoch, step_idx, learning_rate):

    if epoch == args.learning_step[step_idx]:
        learning_rate = learning_rate * 0.1
        step_idx += 1
    return step_idx, learning_rate



def create_folder(args):

    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)
        print("Creat folder: " + args.weights_dir)


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res




def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)



def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


def save_checkpoint(state, args, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.weights_dir, filename))
    if is_best:
        shutil.copy(os.path.join(args.weights_dir, filename),
                    os.path.join(args.weights_dir, 'model_best.pth.tar'))  
