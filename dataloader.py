import torch
import torch.nn as nn
import logging
import torchvision
from torchvision import models
import numpy as np
import torchvision.transforms as transforms

root = "E:\Research\Fb_chall\dataset"
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


#================ Freature Extraction Dataset ==============================
trainset_CIFAR100 = torchvision.datasets.CIFAR100(root, train=True,
                                        download=True, transform=transform)
trainloader_CIFAR100 = torch.utils.data.DataLoader(trainset_CIFAR100, batch_size=64,
                                          shuffle=True)

testset_CIFAR100 = torchvision.datasets.CIFAR100(root, train=False,
                                       download=True, transform=transform)
testloader_CIFAR100 = torch.utils.data.DataLoader(testset_CIFAR100, batch_size=128,
                                         shuffle=False)


#===========================================================================


#================ Classification Dataset ==============================
trainset_CIFAR10 = torchvision.datasets.CIFAR10(root, train=True,
                                        download=True, transform=transform)
trainloader_CIFAR10 = torch.utils.data.DataLoader(trainset_CIFAR10, batch_size=64,
                                          shuffle=True)

testset_CIFAR10 = torchvision.datasets.CIFAR10(root, train=False,
                                       download=True, transform=transform)
testloader_CIFAR10 = torch.utils.data.DataLoader(testset_CIFAR10, batch_size=128,
                                         shuffle=False)
for data in trainloader_CIFAR10:
    print(data)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#===========================================================================