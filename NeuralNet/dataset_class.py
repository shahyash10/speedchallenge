import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler


class DatasetClass(Dataset):

    def __init__(self, img_dir, label_dir=None, transform=None):
        self.img_dir = img_dir
        # skip the first row due to 0th frame
        if label_dir:
            self.labels = np.loadtxt(label_dir, skiprows=1)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img_adr = self.img_dir + str(i) + ".png"
        img = cv2.imread(img_adr)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([self.labels[i]])
