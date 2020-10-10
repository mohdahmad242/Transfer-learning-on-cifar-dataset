import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio


def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1,  bias=True),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True))
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size-delta]

def up_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
    return conv

class encoder(nn.Module):
    def __init__(self, img_ch=1,output_ch=1):
        super(encoder, self).__init__()
       
        #Encoder
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(img_ch, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

    def forward(self, x):
        # encoder
        print(image.size())
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        return x9
        

class decoder(nn.Module):
    def __init__(self, img_ch=1,output_ch=1):
        super(decoder, self).__init__()
       
        #Decoder
        self.up_trans_1 = up_conv(1024, 512)
        self.up_conv_1 = double_conv(1024, 512)
        
        self.up_trans_2 = up_conv(512, 256)
        self.up_conv_2 = double_conv(512, 256)
        
        self.up_trans_3 = up_conv(256, 128)
        self.up_conv_3 = double_conv(256, 128)
        
        self.up_trans_4 = up_conv(128, 64)
        self.up_conv_4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=output_ch,
            kernel_size=1,stride=1,padding=0)  

    def forward(self, image):
        # decoder
        x = self.up_trans_1(image)
        y = crop_img(x, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        x = self.up_trans_2(x)
        y = crop_img(x, x)
        x = self.up_conv_2(torch.cat([x, y], 1))
        
        x = self.up_trans_3(x)
        y = crop_img(x, x)
        x = self.up_conv_3(torch.cat([x, y], 1))
        
        x = self.up_trans_4(x)
        y = crop_img(x, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        
        # output
        x = self.out(x)
        print(x.size())
        return F.softmax(x, dim=0)

from collections import OrderedDict

if __name__ == "__main__":
    image = torch.rand(1, 1, 32, 32)
    en = encoder()
    de = decoder()
    de(en(image))

