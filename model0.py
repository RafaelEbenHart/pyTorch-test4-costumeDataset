# library:
import torch
from torch import nn

import requests
from helper_function import accuracy_fn
from function import evalModel,trainStep,testStep


import torchvision
import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

###########################################################
if Path("Helper_function.py").is_file():
    print("exist")
else:
    print("download")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_function","wb") as f :
        f.write(request.content)

if Path("function.py").is_file():
    print("exist")
else:
    print("download")
    request = requests.get("https://raw.githubusercontent.com/RafaelEbenHart/pyTorch-test3-compVision/refs/heads/main/function.py")
    with open("function","wb") as f:
        f.write(request.content)

device = "cuda" if torch.cuda.is_available else "cpu"

# untuk membuat costume dataset adalah dengan mengggunakan
# costume datasets.

# get Data
# data preparation and data exploration

def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_through_dir("data/photo") <- menampilkan informasi dari folder

# setup train and testing paths
train_dir = "data/photo/train"
test_dir = "data/photo/test"

# Visualizing and image
# 1. Get all of the image path
# 2. pick a random image path using Python random.choice()
# 3. get the image class name using pathlib.Path.parent.stem
# 4. since we're working with images, open image with Python's PIL
# 5. show the image and print metadata

# set seed
# random.seed(42)

# get all of the image path
# note:
# jika pakai manual path sendiri,ubah jadi objek path agar bisa menggunkan glob
image_path = Path("data/photo")
print(image_path)
image_path_list = list(image_path.glob("*/*/*.jpg"))

# print(image_path_list)

# pick a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# get image class from path name
image_class = random_image_path.parent.stem
print(image_class)

# open image
img = Image.open(random_image_path)

# print metadata
print(f"Random Image Path: {random_image_path}")
print(f"Image Class : {image_class}")
print(f"Image Height : {img.height}")
print(f"Image Width : {img.width}")
# img.show()

# visualisasi dengan matplotlib
# turn the image into array

img_as_array = np.array(img)

## plot the image with matplotlib ##
# plt.figure(figsize=(8,5))
# plt.imshow(img_as_array)
# plt.title(f"image class : {image_class} | image shape {img_as_array.shape} => (height, width, color_channel)")
# plt.axis(False)
# plt.show()
# print(img_as_array[1])

# transforming data
# 1. turn your target data into tensor
# 2. turn it into a torch.utils.data.Dataset lalu menjadi  torch.utils.data.Dataloader
# dengan kata lain adalah dataset dan dataLoader

# transforming data with torchvicion.transforms

# untuk menggabungkna transform bisa menggunakan nn.sequential / transforms.compose
data_transform = transforms.Compose([
    # resize images to 64x64
    transforms.Resize(size=(64,64)),
    # flip the images randomly on horizontal untuk data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    # turn the images into torch tensor
    transforms.ToTensor()
])

print(data_transform(img).shape) # mengubah salah satu img ke tensor
# tipe data= float.32

# visualizing data_transform

def plot_transformed_images(image_path: list, transform, n=3, seed=None):
    """
    selectes random image from a path of images
    and loads/trainsform then plots the orginial and trannsformed version
    """
    if seed:
        random.seed(seed)
    random_image_path = random.sample(image_path, k=n)
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig,ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"original\nShape: {f.size}")
            ax[0].axis(False)

            # Transform and plot target images
            transformed_image = transform(f).permute(1, 2, 0) # note : need to change shape for matplotlib to color_channel last
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"transformed\nShape: {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

# plot_transformed_images(image_path=image_path_list,
#                         transform=data_transform,
#                         n=3,
#                         seed=None)
# plt.show()

# option 1: loading image data using torchvision.datasets.ImageFolder
# use imagefolder to create dataset(s)

train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform, # transform for the data
                                  target_transform=None, # transform for the label
                                  )
test_data = datasets.ImageFolder(root=test_dir,
                                transform=data_transform,
                                target_transform=None) # check this for further information may help

# print(f"train data:\n{train_data}")
# print(F"test data:\n{test_data}")

# get class name as a list
class_name = train_data.classes
print(class_name)

# get class name as dict
class_dict = train_data.class_to_idx
print(class_dict)

# check len of data set
print(len(train_data),len(test_data))

# check other use :
# train_data.

# visualizing train_data and test_data
# index on the train_data Dataset to get a single and label
img, label = train_data[0][0],train_data[0][1]
# print(f"image tensor: \n{img}")
print(f"image shape: {img.shape}")
print(f"image datatype: {img.dtype}")
print(f"image labe; : {label}")
print(f"label datatype: {type(label)}")

# Rearrange the order dimensions
img_permute = img.permute(1, 2, 0)
## print out different shape
# print(f"original shape: {img.shape}")
# print(f"permute shape: {img_permute.shape}")

# plot the images
plt.figure(figsize=(8,5))
plt.imshow(img_permute)
plt.axis(False)
plt.title(class_name[label], fontsize = 10)
plt.show()