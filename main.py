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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def load_dataset(dataset = 'cifar10', batch_size = 128):
    data_root = 'data'
    data_root = os.path.abspath(os.path.expanduser(data_root))
    root_dir = os.path.join(data_root, dataset)
    transform_train = transforms.Compose([
                               transforms.Resize(32),
                               transforms.CenterCrop(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

def train(pretrained = False, gen_path = "", disc_path = ""):
    train_data, test_dataset = load_dataset()
    loss_f = torch.nn.BCEWithLogitsLoss()
    epochs = 51
    lr = 1e-5
    G_loss = []
    D_loss = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    generator = Generator()
    discriminator = Discriminator()

    if pretrained:
        generator.load_state_dict(torch.load(gen_path, map_location='cpu'))
        discriminator.load_state_dict(torch.load(disc_path, map_location='cpu'))

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    if torch.cuda.is_available():
      generator.cuda()
      discriminator.cuda()

    print("Parameters of Generator: ", GetNumberParameters(generator))
    print("Parameters of Discriminator: ", GetNumberParameters(discriminator))
    opt_generator = optim.Adam(generator.parameters(), lr = lr)
    opt_discriminator = optim.Adam(discriminator.parameters(), lr = lr)
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        iter = 0
        for im, labels in tqdm(train_data):
            im = im.to(device)
            labels = labels.to(device)
            loss_d = 0
            loss_g = 0
            
            ###################################
            ####### Train Discriminator #######
            ###################################
            # Train with all real images
            discriminator.zero_grad()
            label = torch.full((im.size(0),), 0.9, dtype = torch.float, device = device)
            output = discriminator(im)
            loss_d = loss_f(output.view(-1), label)
            loss_d.backward()

            # Train with all fake images
            inp = torch.randn(im.size(0), 100, 1, 1, device = device)
            fake = generator(inp)
            label.fill_(0.)
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
            label.fill_(0.9)
            output = discriminator(fake)
            loss_g = loss_f(output.view(-1), label)
            loss_g.backward()
            opt_generator.step()

            G_loss.append(loss_g)
            D_loss.append(loss_d)
            if iter % 200 == 0:
              print("G LOSS: ", loss_g.item(), "D LOSS: ", loss_d.item())

            iter += 1
        if epoch % 5 == 0:
            save_model(generator, "gen" + str(datetime.datetime.now().time()) + ".th")
            save_model(discriminator, "disc" + str(datetime.datetime.now().time()) + ".th")

def GenerateArt(model_name, latent_dim = 100):
    from torchvision import transforms

    model = load_model(model_name)
    latent_vector = torch.randn(1, latent_dim, 1, 1)
    art = model(latent_vector)[0, :, :, :]
    print(art)
    # output = art.detach().numpy()
    # output = np.reshape(output, (32, 32, 3))
    # cv2.imwrite('art.png', output)    from torchvision import transforms
    im = transforms.ToPILImage()(art).convert("RGB")
    im = im.resize((1080, 1080))
    im1 = im.save("art.bmp")

    print("Viola! Art is Generated!")

def DisplayArt(path):
    img = cv2.imread(path)
    
    # Displaying the image
    cv2.imshow('image', img)

train()
# GenerateArt('gen17:17:27.803344'+".th")
# DisplayArt('./art.png')
