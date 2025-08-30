import torch
from torch import nn

import requests
from helper_function import accuracy_fn,pred_and_plot_image
from function import train_test_loop,save_results_txt,Save,load,evalModel


import torchvision
import os
import pathlib
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from typing import Tuple,Dict,List
import requests

device = "cuda" if torch.cuda.is_available else "cpu"

train_dir ="data/photo/train"
test_dir ="data/photo/test"

simple_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

train_data_noAug = datasets.ImageFolder(root=train_dir,
                                        transform=simple_transform,
                                        target_transform=None)
test_data_noAug = datasets.ImageFolder(root=test_dir,
                                        transform=simple_transform,
                                        target_transform=None)

train_dataLoader_noAug = DataLoader(dataset=train_data_noAug,
                                    batch_size=8,
                                    shuffle=True,
                                    num_workers=1)
test_dataLoader_noAug = DataLoader(dataset=test_data_noAug,
                                    batch_size=8,
                                    shuffle=False)
class_name = train_data_noAug.classes


class TinyVGGNoAug (nn.Module):
    # ReLU() atau LeakyReLU()
    def __init__(self,input: int,
                    neuron: int,
                    output: int) -> None:
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input,
                        out_channels=neuron,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=neuron,
                        out_channels=neuron,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
                        # stride value pada maxPool2D sama dengan kernel size
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=neuron,
                        out_channels=neuron,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=neuron,
                        out_channels=neuron,
                        kernel_size=3,
                        stride=1,
                        padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=output)
        )
    def forward(self,x):
        return self.classifier(self.convBlock2(self.convBlock1(x)))

torch.manual_seed(42)
loadedModel0 = TinyVGGNoAug(input=3,
                            neuron=10,
                            output=len(train_data_noAug)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
load(model=loadedModel0,Saved_path="models/model0.pth")

custom_image_path = "data/test_image/owi.jpg"

# loading in a custom image
# make custom image is in the same format as data model
# 1. tensor from float32
# 2. shape 64x64x3
# 3. same device

# read in custom image
# custom_image_unit8 = torchvision.io.read_image(costume_image_path)
# print(custom_image_unit8)
# print(custom_image_unit8.shape)
# print(custom_image_unit8.device)
# print(custom_image_unit8.dtype)

# plt.imshow(custom_image_unit8.permute(1, 2, 0))
# plt.show()

## prediciton on a image
# test = evalModel(loadedModel0,test_dataLoader_noAug,loss_fn,accuracy_fn)
# print(test)

custom_image = torchvision.io.read_image(custom_image_path).type(torch.float32) / 255 # dibagi dengan maximum dari color channel
plt.imshow(custom_image.permute(1, 2, 0))
plt.show()
# print(costum_image)

# create transform

custom_transform = transforms.Compose([
    transforms.Resize(size=(64,64))
])

custom_image_transformed = custom_transform(custom_image)
print(custom_image_transformed.shape)

custom_image_transformed = custom_image_transformed.to(device)

loadedModel0.eval()
with torch.inference_mode():
    custom_image_pred = loadedModel0(custom_image_transformed.unsqueeze(0))
    # print(custom_image_pred)
    # print(class_name)
    custom_image_label = torch.softmax(custom_image_pred,dim=1).argmax(dim=1)
    print(class_name[custom_image_label])

pred_and_plot_image(model=loadedModel0,image_path=custom_image_path,transform=custom_transform,device=device,class_names=class_name)
plt.show()
