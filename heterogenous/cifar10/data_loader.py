"""
Create train, valid, test iterators for a chosen dataset.
"""

import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


def data_loader(dataset_name, dataroot, batch_size, val_ratio, world_size, rank):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroor (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation.
        world_size (int): how many processed will be used in training.
        rank (int): the rank of this process.

    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")    

    test_batchsize = 100

    # Mean and std are obtained for each channel from all training images.
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))

    if dataset_name.startswith('CIFAR'):
        # Follows Lee et al. Deeply supervised nets. 2014.
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
    elif dataset_name in ['MNIST', 'FashionMNIST']:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train

    # Load the train dataset.
    train_set = dataset(root=dataroot, train=True,
                        download=True, transform=transform_train)

    # Separates the dataset indices by label.
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train_set):
        class_indices[label].append(idx)

    trainA_classes = [0, 1, 2, 3, 4]
    trainB_classes = [5, 6, 7, 8, 9]

    # Collect/shuffle indices for each loader.
    loaderA_indices = [idx for cls in trainA_classes for idx in class_indices[cls]]
    loaderB_indices = [idx for cls in trainB_classes for idx in class_indices[cls]]
    np.random.shuffle(loaderA_indices)
    np.random.shuffle(loaderB_indices)

    # Load the train datasets.
    train_samplerA = SubsetRandomSampler(loaderA_indices)
    train_loaderA  = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_samplerA,
                              num_workers=1, pin_memory=True)

    train_samplerB = SubsetRandomSampler(loaderB_indices)
    train_loaderB  = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_samplerB,
                              num_workers=1, pin_memory=True)

    # Load the test dataset.
    test_set = dataset(root=dataroot, train=False,
                       download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=test_batchsize,
                             shuffle=False, num_workers=1,
                             pin_memory=True)

    return (train_loaderA, train_loaderB, test_loader)
