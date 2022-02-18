# -*- coding: utf-8 -*-
# author: wuzhuohao
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data


class DCGAN2():
    def __init__(self, image_dir, z_channel, device):
        super(DCGAN2, self).__init__()

        self.train_loader = None
        self.device = device
        self.gen = Generator(z_channel, 3, ngf=64).to(device)
        self.dis = Discriminator(3, ndf=64).to(device)


    def init_dataset(self, image_dir):
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                     std=(0.5, 0.5, 0.5))])
        self.train_set = torchvision.datasets.ImageFolder(image_dir, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set)


    def train_model(self):
        pass











class Generator(nn.Module):
    def __init__(self, z_channel=512, channel=3, ngf=64):
        super(Generator, self).__init__()
        upSampling = []
        in_channel, out_channel = z_channel, ngf * 8
        upSampling += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=1, padding=0),
                       nn.utils.spectral_norm(),
                       nn.ReLU()]

        # using spectral_norm instead of BatchNorm in Generator of original DCGAN

        in_channel = out_channel
        for _ in range(4):
            upSampling += [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                           nn.utils.spectral_norm(),
                           nn.ReLU()]
            in_channel = out_channel
            out_channel /= 2

        upSampling += [nn.ConvTranspose2d(ngf, channel, kernel_size=4, stride=2, padding=1),
                       nn.Tanh()]

        self.upSampling = nn.Sequential(*upSampling)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)  # batch_size, channel, 1x1
        out = self.upSampling(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, channel=3, ndf=64):
        super(Discriminator, self).__init__()
        DownSampling = []
        DownSampling += [nn.Conv2d(channel, ndf, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.utils.spectral_norm(),
                         nn.LeakyReLU(0.2, inplace=True)]

        in_channel, out_channel = ndf, ndf
        for i in range(3):
            in_channel = out_channel
            out_channel *= 2
            DownSampling += [nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
                             nn.utils.spectral_norm(),
                             nn.LeakyReLU(0.2, inplace=True)]

        in_channel = out_channel
        DownSampling += [nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False),
                         nn.utils.spectral_norm(),
                         nn.LeakyReLU(0.2, inplace=True)]

        DownSampling += [nn.Conv2d(in_channel, 1, kernel_size=4, stride=1, padding=0, bias=False),
                         nn.LeakyReLU(0.2, inplace=True)]

        self.DownSampling = nn.Sequential(*DownSampling)

    def forward(self, x):
        out = self.DownSampling(x).view(-1, 1)
        return out

