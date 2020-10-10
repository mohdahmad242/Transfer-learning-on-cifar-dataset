from dataloader import trainloader_CIFAR100,testloader_CIFAR100, trainloader_CIFAR10, testloader_CIFAR10, classes
import torchvision
from unet import UNet
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


#============== Model ================

unet = UNet()


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader_CIFAR10, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    val_loss = 0
    with torch.no_grad():
        for data in testloader_CIFAR10:
            images, labels = data
            outputs = unet(images)
            loss = criterion(outputs, outputs)
            val_loss += loss.item()

    print("{} Epoch - Traningloss - {}".format(epoch+1, running_loss/len(trainloader_CIFAR10)))
    print("Validationloss - {}".format(val_loss/len(testloader_CIFAR10)))

print('Finished Training')