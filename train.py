from dataloader import trainloader_CIFAR100,testloader_CIFAR100, trainloader_CIFAR10, testloader_CIFAR10, classes
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


#================ Freature Extraction Model ==============================
from unet import UNet
unet = UNet()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(unet.parameters(), lr=0.001, momentum=0.9)

def train_unet():
    for epoch in range(1): 

        running_loss = 0.0
        for i, data in enumerate(trainloader_CIFAR100, 0):
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
            for data in testloader_CIFAR100:
                images, labels = data
                outputs = unet(images)
                loss = criterion(outputs, outputs)
                val_loss += loss.item()

        print("{} Epoch - Traningloss - {}".format(epoch+1, running_loss/len(trainloader_CIFAR10)))
        print("Validationloss - {}".format(val_loss/len(testloader_CIFAR10)))
    torch.save(unet.state_dict(), "./model/unet.pth")

#================ Classification Model ==============================

from classifier import encoder, decoder
encoder = encoder()
try:
    preTrained_unet = torch.load("./model/unet.pth")
    encoder.load_state_dict(preTrained_unet, strict = False)
decoder = decoder()


def train_classifier():
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader_CIFAR10, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = encoder(inputs)
            outputs = decoder(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        val_loss = 0
        with torch.no_grad():
            for data in testloader_CIFAR10:
                images, labels = data
                outputs = encoder(inputs)
                outputs = decoder(outputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print("{} Epoch - Traningloss - {}".format(epoch+1, running_loss/len(trainloader_CIFAR10)))
        print("Validationloss - {}".format(val_loss/len(testloader_CIFAR10)))

train_unet()
train_classifier()
print('Finished Training')