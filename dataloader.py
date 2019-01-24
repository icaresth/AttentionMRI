import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from skimage import transform

import numpy as np
import pandas as pd
import nibabel as nib

import os

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

########################################################################################################################
# read '.nii' files
def nii_loader(file):
    data = nib.load(file)
    img = data.get_data()
    return img


# dataset class
class ADHDDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        idx = int(idx)
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0][1: -1] + '.nii')

        data = nii_loader(img_name)
        label = self.csv_file.iloc[idx, 5]
        if label >= 1:
            label = 1

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

########################################################################################################################
# transforms
# spatial resize
class resize(object):
    def __init__(self, size):
        assert isinstance(size, (tuple))
        self.size = size

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        da = transform.resize(data, self.size)
        new_sample = {'data': da, 'label': label}
        return new_sample

class normalization(object):
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data -= np.mean(data)
        data /= np.std(data)
        new_sample = {'data': data, 'label': label}
        return new_sample

class normalization2(object):
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        max = np.max(data)
        min = np.min(data)
        delta = max - min
        d = (data - min) / delta
        new_sample = {'data': d, 'label': label}
        return new_sample

class crop(object):
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        da = data[15: 105, 14: 131, 7: 97]
        new_sample = {'data': da, 'label': label}
        return new_sample

class cvt2d(object):
    def __init__(self, direction):
        assert isinstance(direction, (str))
        self.dir = direction

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        x, y, h = data.shape

        if self.dir == 'c': # coronal (y,| x, h)
            d = data[:, y//2, :]
        elif self.dir == 's': # sagittal (x,| y, h)
            d = data[x//2, :, :]
        elif self.dir == 'a': # asials (h,| x, y)
            d = data[:, :, h//2]
        else:
            d = data[:, y//2, :]

        new_sample = {'data': d, 'label': label}
        return new_sample


class ToTensor(object):
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        d = []
        for i in range(1):
            d.append(data)

        d = np.asarray(d, dtype=np.float32)
        d = torch.from_numpy(d)

        l = np.asarray(label, dtype=np.longlong)
        l = torch.from_numpy(l)

        new_sample = {'data': d, 'label': l}
        return new_sample


########################################################################################################################
# dataset and dataloader
wm_data_dir = '/media/kamata-1080/1b32372a-0e1c-4f6f-9681-7682b27049a41/MyWork/my_work_new/data/wm/train'
csv_dir = '/media/kamata-1080/1b32372a-0e1c-4f6f-9681-7682b27049a41/MyWork/my_work_new/cae.csv'

data_dir = '/media/kamata-1080/1b32372a-0e1c-4f6f-9681-7682b27049a41/MyWork/my_work_new/data/gm'
gm_train_csv = "/media/kamata-1080/1b32372a-0e1c-4f6f-9681-7682b27049a41/MyWork/dataset/training.csv"
gm_test_csv = "/media/kamata-1080/1b32372a-0e1c-4f6f-9681-7682b27049a41/MyWork/dataset/test.csv"

data_transforms = {
    'train': transforms.Compose([
        crop(),
        resize((96, 128, 96)),
        ToTensor()
    ]),
    'test': transforms.Compose([
        crop(),
        resize((96, 128, 96)),
        ToTensor()
    ])
}

data_label = {
    'train': gm_train_csv,
    'test': gm_test_csv
}


image_datasets = {x: ADHDDataset(csv_file=data_label[x], root_dir=os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'test']}


def get_train_valid_loader(batch_size, valid_size=0.2, shuffle=True, num_workers=4):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_train = len(image_datasets['train'])
    split = int(np.floor(valid_size * num_train))
    image_datasets['train'], image_datasets['val'] = random_split(image_datasets['train'], [num_train - split, split])

    t = len(image_datasets['train'])
    v = len(image_datasets['val'])

    train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    valid_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (train_loader, valid_loader, t, v)


def get_test_loader(batch_size, num_workers=4):
    dataset = image_datasets['test']
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader