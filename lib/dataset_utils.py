'''
Dataset and DataLoader adapted from
https://www.kaggle.com/pinocookie/pytorch-dataset-and-dataloader
'''

import pickle

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler


def load_mnist(batch_size,
               data_dir='./data',
               val_size=0.1,
               shuffle=True,
               seed=1):
    """Load MNIST data into train/val/test data loader"""

    num_workers = 4

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    validset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def load_mnist_all(data_dir='./data', val_size=0.1, shuffle=True, seed=1):
    """Load entire MNIST dataset into tensor"""

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    validset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=(num_train - split), sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=split, sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    x_train = next(iter(trainloader))
    x_valid = next(iter(validloader))
    x_test = next(iter(testloader))

    return x_train, x_valid, x_test


def load_cifar10(batch_size,
                 data_dir='./data',
                 val_size=0.1,
                 augment=True,
                 shuffle=True,
                 seed=1):
    """Load CIFAR-10 data into train/val/test data loader"""

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_workers = 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform_train = transform

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train)
    validset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    # Random split train and validation sets
    num_train = len(trainset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader


def load_gtsrb(data_dir='./data', gray=False, train_file_name=None):
    """
    Load GTSRB data as a (datasize) x (channels) x (height) x (width) numpy
    matrix. Each pixel is rescaled to lie in [0,1].
    """

    def load_pickled_data(file, columns):
        """
        Loads pickled training and test data.

        Parameters
        ----------
        file    : string
                          Name of the pickle file.
        columns : list of strings
                          List of columns in pickled data we're interested in.

        Returns
        -------
        A tuple of datasets for given columns.
        """

        with open(file, mode='rb') as f:
            dataset = pickle.load(f)
        return tuple(map(lambda c: dataset[c], columns))

    def preprocess(x, gray):
        """
        Preprocess dataset: turn images into grayscale if specified, normalize
        input space to [0,1], reshape array to appropriate shape for NN model
        """

        if not gray:
            # Scale features to be in [0, 1]
            x = (x / 255.).astype(np.float32)
        else:
            # Convert to grayscale, e.g. single Y channel
            x = 0.299 * x[:, :, :, 0] + 0.587 * x[:, :, :, 1] + \
                0.114 * x[:, :, :, 2]
            # Scale features to be in [0, 1]
            x = (x / 255.).astype(np.float32)
            x = x[:, :, :, np.newaxis]
        return x

    # Load pickle dataset
    if train_file_name is None:
        x_train, y_train = load_pickled_data(
            data_dir + 'train.p', ['features', 'labels'])
    else:
        x_train, y_train = load_pickled_data(
            data_dir + train_file_name, ['features', 'labels'])
    x_val, y_val = load_pickled_data(
        data_dir + 'valid.p', ['features', 'labels'])
    x_test, y_test = load_pickled_data(
        data_dir + 'test.p', ['features', 'labels'])

    # Preprocess loaded data
    x_train = preprocess(x_train, gray)
    x_val = preprocess(x_val, gray)
    x_test = preprocess(x_test, gray)
    return x_train, y_train, x_val, y_val, x_test, y_test


class GtsrbDataset(torch.utils.data.Dataset):

    def __init__(self, x_np, y_np, mean=None, std=None, augment=False):

        self.x_pil = [Image.fromarray(
            (x * 255).astype(np.uint8)) for x in x_np]
        self.y_np = y_np.astype(np.int64)

        if mean is None:
            mean = (0, 0, 0)
            std = (1, 1, 1)

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='edge'),
                transforms.RandomAffine(
                    5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
                transforms.ColorJitter(brightness=0.1),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ])

    def __getitem__(self, index):
        # apply the transformations and return tensors
        return self.transform(self.x_pil[index]), self.y_np[index]

    def __len__(self):
        return len(self.x_pil)


def load_gtsrb_dataloader(data_dir, batch_size, num_workers=4):

    x_train, y_train, x_val, y_val, x_test, y_test = load_gtsrb(
        data_dir=data_dir)

    # Standardization
    mean = np.mean(x_train, (0, 1, 2))
    std = np.std(x_train, (0, 1, 2))

    trainset = GtsrbDataset(x_train, y_train, mean, std, augment=True)
    validset = GtsrbDataset(x_val, y_val, mean, std, augment=False)
    testset = GtsrbDataset(x_test, y_test, mean, std, augment=False)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader
