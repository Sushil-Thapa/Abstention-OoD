"""
a data loader for the semi-supervised training using CIFAR-100 and tiny images.
Loosely based on the CIFAR-10 Dataset class of PyTorch
"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
#from .utils import download_url, check_integrity


class TinyImage(data.Dataset):

    def __init__(self, input, target, train=True,
                 transform=None, target_transform=None):
                 
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.num_inputs = 1
        self.num_targets = 1
        if self.train:
            self.train_data = input
            self.train_labels = target
            self.num_train = self.train_data.shape[0]
            self.train_data = self.train_data.reshape((self.num_train, 3, 32, 32))
            # self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_data = self.train_data.transpose((0, 3, 2, 1))  # convert to HWC

        else:
            self.test_data = input
            self.test_labels = target
            self.num_test = self.test_data.shape[0]
            self.test_data = self.test_data.reshape((self.num_test, 3, 32, 32))
            # self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_data = self.test_data.transpose((0, 3, 2, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


