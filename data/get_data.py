import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from util.cutout import Cutout


def get_data(batch_size=256, dataset='cifar10'):
    if dataset == 'cifar10':
        return get_data_cifar10(batch_size=batch_size)
    elif dataset == 'cifar100':
        return get_data_cifar100(batch_size=batch_size)
    elif dataset == 'imagenet':
        return get_data_imagenet(batch_size=batch_size)
    elif dataset == 'mnist':
        return get_data_mnist(batch_size=batch_size)
    elif dataset == 'tiny':
        return get_data_tinyimagenet(batch_size=batch_size)


def get_data_cifar10(batch_size=256):
    tv_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
        tv_normalize,
        Cutout(n_holes=1, length=16),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        tv_normalize
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader


def get_data_imagenet(batch_size=32):
    tv_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomCrop(32, 4),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        tv_normalize
    ])
    val_transform = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
        tv_normalize
    ])
    # trainset = torchvision.datasets.ImageNet(root='/home/faker1907/INFOCOM/data/imagenet_val',split="val",
    #                                         download=True, transform=train_transform)
    # testset = torchvision.datasets.ImageNet(root='/home/faker1907/INFOCOM/data/imagenet',
    #                                        download=True, transform=val_transform)
    train_dataset = torchvision.datasets.ImageFolder(
        root='/home/faker1907/INFOCOM/data/imagenet',
        transform=train_transform)
    train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=20)

    val_dataset = torchvision.datasets.ImageFolder(
        root='/home/faker1907/INFOCOM/data/imagenet',
        transform=val_transform)

    val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=20)

    return train_dataset, train_dataset_loader, val_dataset, val_dataset_loader


def get_data_cifar100(batch_size=32):
    tv_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
        tv_normalize,
        Cutout(n_holes=1, length=16),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        tv_normalize
    ])
    root_path = './data/cifar100'
    trainset = torchvision.datasets.CIFAR100(root=root_path, train=True,
                                             download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=root_path, train=False,
                                            download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def get_data_mnist(batch_size=32):
    tv_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    root_path = './data/mnist'
    trainset = torchvision.datasets.MNIST(root=root_path, train=True,
                                          download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.MNIST(root=root_path, train=False,
                                         download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainset, trainloader, testset, testloader


def get_data_tinyimagenet(batch_size, data_dir='./data/tiny-imagenet-200'):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         torchvision.transforms.RandomCrop(64, 4),
         transforms.ToTensor(),
         normalize,
         # Cutout(n_holes=1, length=16),
         ])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return trainset, train_loader, testset, test_loader
