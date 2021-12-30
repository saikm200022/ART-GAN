'''
GANs can be used to create new data that follows some distribution. 

Generator - produce images that look very close to training images to trick Discriminator

Discriminator - look at produced image and determine authenticity

Goal is to generate perfect images that look very close to the training data, causing the Discrminator
to guess at 50% confidence (random chance)

'''

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
from Generator import *
from Discriminator import *

def load_dataset(dataset = 'cifar10'):
    data_root = 'data'
    data_root = os.path.abspath(os.path.expanduser(data_root))
    root_dir = os.path.join(data_root, dataset)
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ColorJitter(brightness=0.5, hue = 0.25),
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Use Torchvision to load CIFAR10 Dataset
    train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset

def train():
    train_data, test_dataset = load_dataset()
    loss_f = torch.nn.BCELoss()
    epochs = 10
    labels = [0, 1]

    generator = Generator()
    discriminator = Discriminator()
    opt_generator = optim.Adam(generator.parameters(), lr = 1e-2)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr = 1e-2)
    for epoch in range(epochs):
        for im, labels in enumerate(train_data):
            ###################################
            ####### Train Discriminator #######
            ###################################
            # Train with all real images
            discriminator.zero_grad()
            label = torch.full((im.size(0)), 1)
            output = discriminator(im)
            loss_d = loss_f(output, label)
            loss_d.backward()

            # Train with all fake images
            input = torch.rand(im.size(0), 32, 32)
            fake = generator(input)
            label.fill_(0)
            output = discriminator(input)
            loss_fake = loss_f(output, label)
            loss_fake.backward()

            # Accumulate Loss
            loss_d += loss_fake
            opt_discriminator.step()

            
            