import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import scipy
import numpy as np
import time
import os


def img_loader(path):
    return Image.open(path).convert('RGB')


def list_reader(flist):

    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.split()
            imlist.append((impath, int(imlabel)))
								
    return imlist


class MyDataset(data.dataset.Dataset):
    def __init__(self, flist, transform=None):
    	
    	self.imlist = list_reader(flist)
    	self.transform = transform

    def __getitem__(self, idx):

        impath, target = self.imlist[idx]
        img = img_loader(os.path.join('/data/val', impath))
        if self.transform is not None:
            img = self.transform(img)

        return img, target


    def __len__(self):
        return len(self.imlist)

