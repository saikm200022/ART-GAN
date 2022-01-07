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
from torchvision.utils import save_image

from torchvision import transforms

def grid_view(model_name, latent_dim):
    model = Generator()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    latent_vector = torch.randn(128, latent_dim, 1, 1)
    art = model(latent_vector)
    plt.figure(figsize=(32,32))
    plt.axis("off")
    plt.title("Examples of Generated Data")
    plt.imshow(np.transpose(vutils.make_grid(art.to(device), padding = 0, normalize=True).cpu(),(1,2,0)))
    plt.show()

def create_data(model_name, latent_dim, num_images = 16, dir = "./additional_data"):
    os.mkdir(dir)
    model = Generator()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    for i in range(num_images):
        latent_vector = torch.randn(128, latent_dim, 1, 1)
        art = model(latent_vector)
        save_image(art[0], './' + dir + '/' + str(i + 1) + '.jpeg')
    print("Voila! Additional Data is Generated!")


model_name = './gen490'+".th"
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# grid_view(model_name = './gen490.th', latent_dim = 100)
create_data(model_name, latent_dim, num_images = 128, dir = "./generated_data")
