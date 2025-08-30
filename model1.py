# library:
import torch
from torch import nn

import requests
from helper_function import accuracy_fn
from function import evalModel,train_test_loop
from poltFunction import display_random_image,plot_loss_curves


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
from torchinfo import summary

def main():

    device = "cuda" if torch.cuda.is_available else "cpu"

    train_dir = "data/photo/train"
    test_dir = "data/photo/test"

    train_data_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_data_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                    transform=train_data_transform,
                                    target_transform=None,
                                        )
    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=test_data_transform,
                                    target_transform=None)

    class_name = train_data.classes
    # print(class_name)

    BATCH_SIZE = 8
    # NUM_WORKERS = os.cpu_count()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # print(NUM_WORKERS)
    train_dataLoader = DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=1)
    # num_worker bia di set menjadi os.cpu_count()
    # hal ini akan set num_workers sebanyak mungkin
    test_dataLoader = DataLoader(dataset=test_data,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=1)

    # model1 : with data augmentation
    class TinyVGGWAug (nn.Module):
        def __init__(self, input:int,
                     neurons:int,
                     output:int)->None:
            super().__init__()
            self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=input,
                          out_channels=neurons,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=neurons,
                          out_channels=neurons,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=neurons,
                          out_channels=neurons,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=neurons,
                          out_channels=neurons,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(out_features=output)
            )
        def forward(self, x):
            return self.classifier(self.convblock2(self.convblock1(x)))

    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    model1 = TinyVGGWAug(input=3,
                         neurons=10,
                         output=len(train_data.classes)).to(device)

    # loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model1.parameters(),
                                 lr=1e-3)

    model1_results = train_test_loop(model=model1,
                                     lossFn=loss_fn,
                                     optimizer=optimizer,
                                     train_dataLoader=train_dataLoader,
                                     test_dataLoader=test_dataLoader,
                                     perBatch=None,
                                     epochs=5)
    plot_loss_curves(results=model1_results)
    plt.show()


if __name__ == "__main__" :
    main()