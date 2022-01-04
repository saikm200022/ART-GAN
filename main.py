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
from tqdm import tqdm
import datetime
import cv2

def load_dataset(dataset = 'cifar10', batch_size = 32):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def save_model(model, file_name):
    from torch import save
    from os import path
    print("saving", file_name)
    return save(model.state_dict(), file_name)


def load_model(file):
    from torch import load
    from os import path
    gen = Generator()
    gen.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), file), map_location='cpu'))
    return gen

def GetNumberParameters(model):
  return sum(np.prod(p.shape).item() for p in model.parameters())

def train():
    train_data, test_dataset = load_dataset()
    loss_f = torch.nn.BCELoss()
    epochs = 10
    G_loss = []
    D_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    generator = Generator()
    discriminator = Discriminator()
    print("Parameters of Generator: ", GetNumberParameters(generator))
    print("Parameters of Discriminator: ", GetNumberParameters(discriminator))
    opt_generator = optim.Adam(generator.parameters(), lr = 1e-2)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr = 1e-2)
    for epoch in range(epochs):
        for im, labels in tqdm(train_data):
            im = im.to(device)
            labels = labels.to(device)

            ###################################
            ####### Train Discriminator #######
            ###################################
            # Train with all real images
            discriminator.zero_grad()
            label = torch.full((im.size(0),), 1, dtype = torch.float, device = device)
            output = discriminator(im)
            loss_d = loss_f(output.view(-1), label)
            loss_d.backward()

            # Train with all fake images
            inp = torch.rand(im.size(0), 100, 1, 1, device = device)
            fake = generator(inp)
            label.fill_(0)
            output = discriminator(fake.detach())
            loss_fake = loss_f(output.view(-1), label)
            loss_fake.backward()

            # Accumulate Loss
            loss_d += loss_fake
            opt_discriminator.step()

            ##################################
            ###### Train Generator ###########
            ##################################
            generator.zero_grad()
            label.fill_(1)
            output = discriminator(fake)
            loss_g = loss_f(output.view(-1), label)
            loss_g.backward()
            opt_generator.step()

            G_loss.append(loss_g)
            D_loss.append(loss_d)

        if epoch % 5 == 0:
            save_model(generator, "gen" + str(datetime.datetime.now().time()) + ".th")
            save_model(discriminator, "disc" + str(datetime.datetime.now().time()) + ".th")

def GenerateArt(model_name, latent_dim = 100):
    from torchvision import transforms

    model = load_model(model_name)
    latent_vector = torch.randn(1, latent_dim, 1, 1)
    art = model(latent_vector)[0, :, :, :]
    output = art.detach().numpy()
    output = np.reshape(output, (32, 32, 3))
    cv2.imwrite('art.png', output)
    print("Viola! Art is Generated!")

def DisplayArt(path):
    img = cv2.imread(path)
    
    # Displaying the image
    cv2.imshow('image', img)

# train()
# GenerateArt('gen12:25:44.582487'+".th")
# DisplayArt('./art.png')
