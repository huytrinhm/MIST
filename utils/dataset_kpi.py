import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class KPIsDataset(Dataset):
    def __init__(self, root_dir, split='train', image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        image_directory_path = os.path.join(root_dir, split, 'img')
        mask_directory_path = os.path.join(root_dir, split, 'mask')

        image_filenames = [filename for filename in os.listdir(image_directory_path) if filename.endswith('.pt')]
        mask_filenames = [filename for filename in os.listdir(mask_directory_path) if filename.endswith('.pt')]
        assert len(image_filenames) == len(mask_filenames)

        self.image_paths = [os.path.join(image_directory_path, filename) for filename in image_filenames]
        self.mask_paths = [os.path.join(mask_directory_path, filename[:-11] + '_mask.jpg.pt') for filename in image_filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.load(self.image_paths[idx])
        mask = torch.load(self.mask_paths[idx])

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {'image': image, 'label': mask}
        
        sample['case_name'] = self.image_paths[idx]
        return sample

class KPIsTestDataset(Dataset):
    def __init__(self, root_dir, image_transform=None):
        self.image_transform = image_transform

        image_filenames = [filename for filename in os.listdir(root_dir) if filename.endswith('.pt')]

        self.image_paths = [os.path.join(root_dir, filename) for filename in image_filenames]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.load(self.image_paths[idx])

        if self.image_transform:
            image = self.image_transform(image)

        sample = {'image': image}
        
        sample['case_name'] = self.image_paths[idx]
        return sample
