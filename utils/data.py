import os
import numpy as np
from .autoaugment import ImageNetPolicy
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    def __init__(self):
        self.train_trsf = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63/255),
            ImageNetPolicy(),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        self.common_trsf = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        
        self.data_dir = ...
        self.dataset_name = None
        self.class_order = None

    def download_data(self):
        data_dir = f'{self.data_dir}/{self.dataset_name}'
        assert os.path.exists(data_dir), f'Data folder {data_dir} not found.'
        train_dir = f'{data_dir}/train'
        test_dir = f'{data_dir}/test'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iCaltech256(iData):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Caltech256'
        self.class_order = np.arange(256).tolist()


class iFood101(iData):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Food101'
        self.class_order = np.arange(101).tolist()


class iPlaces100(iData):
    def __init__(self):
        super().__init__()
        self.dataset_name = 'Places100'
        self.class_order = np.arange(100).tolist()

