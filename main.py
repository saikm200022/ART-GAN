'''

Main entrypoint into code that can display generated images in grid format as well as 
a utility function that can create a directory and save generated data images.

'''

import os
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from architecture.Generator import *
from architecture.Discriminator import *
from torchvision.utils import save_image

def grid_view(model_name, latent_dim):
    model = Generator()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    latent_vector = torch.randn(128, latent_dim, 1, 1)
    output = model(latent_vector)
    plt.figure(figsize=(32,32))
    plt.axis("off")
    plt.title("Examples of Generated Data")
    plt.imshow(np.transpose(vutils.make_grid(output.to(device), padding = 0, normalize=True).cpu(),(1,2,0)))
    plt.show()

def create_data(model_name, latent_dim, num_images = 16, dir = "./additional_data"):
    os.mkdir(dir)
    model = Generator()
    model.load_state_dict(torch.load(model_name, map_location='cpu'))

    for i in range(num_images):
        latent_vector = torch.randn(128, latent_dim, 1, 1)
        output = model(latent_vector)
        save_image(output[0], './' + dir + '/' + str(i + 1) + '.jpeg')
    print("Voila! Additional Data is Generated!")

model_name = './model_checkpoints/generator'+".th"
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

grid_view(model_name, latent_dim = 100)
# create_data(model_name, latent_dim, num_images = 128, dir = "./generated_data")
