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
img.show()
