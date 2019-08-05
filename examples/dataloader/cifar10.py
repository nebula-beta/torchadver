#!/usr/bin/python2.7
#coding:utf-8

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_train_loader(batch_size=128, num_workers=8, transform=None, shuffle=True):
    if transform is None:

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    cifar10_dataset = datasets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader


def get_val_loader(batch_size=128, num_workers=8, transform=None):

    if transform is None:

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    cifar10_dataset = datasets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader

get_test_loader = get_val_loader

if __name__ == '__main__':
    get_train_loader()
    get_val_loader()
