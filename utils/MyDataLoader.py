'''

Template code to build a pytorch dataloader using the new generated data.

'''

import torch
import torchvision.transforms
import os 
import numpy as np
from torchvision import datasets, transforms
import cv2
from PIL import Image


class MyDataLoader(torch.utils.data.Dataset):
    def __init__(self):
        directory = './generated_data'
        self.dataset = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpeg"):
                im = cv2.imread("./generated_data" + filename)
                self.dataset.append(im)
                
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        pil = Image.fromarray(sample)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return transform(pil)

    def __len__(self):
        return len(self.dataset)