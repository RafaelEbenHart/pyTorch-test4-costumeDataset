# library:
import torch
from torch import nn

import requests
from helper_function import accuracy_fn
from function import train_test_loop,save_results_txt,Save


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

###########################################################
## option 2: loading Image data with a costume Dataset
# 1. able to load image from file
# 2. able to get class name from the Dataset
# 3. able to get classes as dictionary from the dataset

# pros:
# 1. can create a Dataset out of almost anything
# 2. Not Limited ro Pytorch pre-built Dataset functions

# cons:
# 1. even though you could create dataset out of almost everything,it doesnt mean it'll work
# 2. using a costume Dataset often result in writing more code,which could be prone to errors or performance issue

# instance of torchVision.dataset.ImageFolder()
# print(train_data.classes, train_data.class_to_idx)

# create helper function to get class names like ImageFolder()
# step:
# 1. get the class names using os.scandir() to traverse a target directory
#    (idealy the directory is in standrad image classification format)
# 2. Raise an error if the class name arent found (if this happens,there might be
#    something wrong with the directory structure)
# 3. turn the class cames into a dict and a list

# setup path for target directory
target_directory = "data/photo/train"
# print(f"target dir: {target_directory}")
# get the class name from the target directory
class_name_found = sorted([entry.name for entry in list(os.scandir(target_directory))])
# print(class_name_found)

# make a class for find classes
def find_classes(directory: str) -> Tuple[list[str], Dict[str,int]]:
    """
    Finds the class folder name in a target directory
    """
    # 1. get the class name by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # print(classes)

    # 2. raise an error if class name could not be found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}..")

    # 3. Create a dictionary of index labels (computer prefer numbers rather then str as labels)
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    # print(class_to_idx)
    return classes, class_to_idx

find_classes(target_directory)

# create a costume Dataset to replicate ImageFolder

# 1. sublass torch.utils.data.Dataset
# 2. init our subclass with a target directory (the directory like to get data from)
#   as well as a transform if like to transform data
# 3. Create several attributes:
#   a. paths - path of images
#   b. transform - a list of the target classes
#   c. classes - a list of the target classes
#   d. class_to_idx - a dict of target classes mapped to integer labels
# 4. Create a function to load_images(), this function will open an image
# 5. overwrite the __len()__ method to return the length of dataset
# 6. overwrite the __getitem()__ method to return a given sample when passed an index

# write a costume dataset class
# 1.
class imageFolderCustom(Dataset):
    # 2.
    def __init__(self, targ_dir: str,
                    transform: None):
        # 3.
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transforms = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

        # 4.
    def load_image(self, index:int) -> Image.Image:
        "opens an image via a path and returns it"
        image_path = self.paths[index]
        return Image.open(image_path)

        # 5.
    def __len__(self) -> int:
        "return the total number of samples"
        return len(self.paths)

        # 6.
    def __getitem__(self, index:int)->Tuple[torch.Tensor, int]:
        # __getitem__ akan mereplikasi:
        # img,lkabel = train_data[0] -> [img, label]
        #              ↑↑↑↑↑↑↑↑↑↑↑↑↑
        "returns one sample of data, data and label (X,y)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expect path in format: data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # transform is necessary
        if self.transforms:
            return self.transforms(img), class_idx # return data ,label (X,y)
        else:
            return img,class_idx # return untransformed image and label

###########################################################
def main():
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
    # print(image_path)
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # print(image_path_list)

    # pick a random image path
    random_image_path = random.choice(image_path_list)
    # print(random_image_path)

    # get image class from path name
    image_class = random_image_path.parent.stem
    # print(image_class)

    # open image
    img = Image.open(random_image_path)

    # print metadata
    # print(f"Random Image Path: {random_image_path}")
    # print(f"Image Class : {image_class}")
    # print(f"Image Height : {img.height}")
    # print(f"Image Width : {img.width}")
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

    # print(data_transform(img).shape) # mengubah salah satu img ke tensor
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
    # print(class_name)

    # get class name as dict
    class_dict = train_data.class_to_idx
    # print(class_dict)

    # check len of data set
    # print(len(train_data),len(test_data))

    # check other use :
    # train_data.

    # visualizing train_data and test_data
    # index on the train_data Dataset to get a single and label
    img, label = train_data[0][0],train_data[0][1]
    # print(f"image tensor: \n{img}")
    # print(f"image shape: {img.shape}")
    # print(f"image datatype: {img.dtype}")
    # print(f"image label : {label}")
    # print(f"label datatype: {type(label)}")

    # Rearrange the order dimensions
    img_permute = img.permute(1, 2, 0)
    ## print out different shape
    # print(f"original shape: {img.shape}")
    # print(f"permute shape: {img_permute.shape}")

    ## plot the images
    # plt.figure(figsize=(8,5))
    # plt.imshow(img_permute)
    # plt.axis(False)
    # plt.title(class_name[label], fontsize = 10)
    # plt.show()

    # dataLoader

    BATCH_SIZE = 16
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

    # print(len(train_dataLoader),len(test_dataLoader))
    # print(len(train_data), len(test_data))

    img,label =  next(iter(train_dataLoader))
    # batch size is 8:
    # print(f"Image shape: {img.shape} -> (batch_size, color_channel, height, width)")
    # print(f"Label shape: {label.shape}")


    # create a transform
    train_transforms = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor()
    # pada test data biasanya tidak menggunakan data augmentation
])

    # test out imageFolderCustom
    train_data_custom = imageFolderCustom(targ_dir=train_dir,
                                            transform=train_transforms)
    test_data_custom = imageFolderCustom(targ_dir=test_dir,
                                           transform=test_transforms)
    # print out
    # print(len(train_data_costume), len(train_data))
    # print(len(test_data_costume), len(test_data))
    # print(train_data_costume.classes)
    # print(train_data_costume.class_to_idx)

    # check for equality between orginal and custom

    # print(train_data_costume.classes == train_data.classes)
    # print(test_data_costume.classes == test_data.classes)

    ## visualizing function (ImageFolder)
    # display_random_image(train_data,
    #                      n=5,
    #                      classes=class_name,
    #                      seed=None)

   # visualizing function (ImageFolderCustom)
    # display_random_image(train_data_custom,
    #                      n=20,
    #                      classes=class_name,
    #                      seed=42)

    # turn custom loaded images into dataloader again

    train_custom_dataLoader = DataLoader(dataset=train_data_custom,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=1)

    test_custom_dataLoader = DataLoader(dataset=test_data_custom,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=1)
    # get image and label from custom dataloader
    img_custom, label_custom = next(iter(train_custom_dataLoader))
    # print(img_custom.shape,label_custom.shape)

    # other forms of transform (data augmentation)
    # data augmentation adalah menerapkan keberagaman pada training data
    # hal ini bertujuan untuk menambah jumlah variasi pada gambar

    # trivial Augment
    train_transforms = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor()
    ])

    # get all image paths
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    # print(image_path_list[:10])

    aug_data = [train_transforms(Image.open(p)) for p in image_path_list]
    len_aug_data = len(aug_data)
    len_data = len(train_data_custom)
    print(f"jumlah augmentasi: {len_aug_data} gambar")
    print(f"jumlah train data: {len_data} gambar")
    print(f"jumlah train data + augmentasi: {len_data + len_aug_data} gambar")


    # plot random transform images
    # plot_transformed_images(image_path=image_path_list,
    #                         transform=train_transforms,
    #                         n=3,
    #                         seed=None)
    # plt.show()

    # model0 : tinyVGG without data augmentation

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
                                       shuffle=False,
                                       num_workers=1)
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
    model0 = TinyVGGNoAug(input=3,
                          neuron=10,
                          output=len(class_name)).to(device)
    # torch.manual_seed(42)
    # dummyData = torch.randn(8, 3, 64, 64)
    # dummyData = dummyData.to(device)
    # print(model0(dummyData))

    # get a single image batch
    image_batch, label_batch = next(iter(train_dataLoader_noAug))
    print(image_batch.shape, label_batch.shape)

    # try a forward pass
    image_batch = image_batch.to(device)
    print(model0(image_batch))

    # print(summary(model0,input_size=(8, 3, 64, 64)))

    # loss_function and optimizer

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model0.parameters(),
                                lr=1e-3 )

    # Train step and test step loop
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model0_result=train_test_loop(model=model0,
                                  lossFn=loss_fn,
                                  optimizer=optimizer,
                                  train_dataLoader=train_dataLoader_noAug,
                                  test_dataLoader=test_dataLoader_noAug,
                                  perBatch=None,
                                  epochs=5)

    ## plot loss curves of model
    # tracking  model progress overtime

    def plot_loss_curves(results: Dict[str,List[float]]):
        """Plots training curves of a results dictionary"""
        # get the loss values of the results dictionary(train and test)
        train_loss = results["train_loss"]
        test_loss = results["test_loss"]
        # get the acc values of the results dictionary(train and test)
        train_acc = results["train_acc"]
        test_acc = results["test_acc"]
        # figure out how many epochs
        epochs = range(len(results["train_loss"]))
        # setup plot
        plt.figure(figsize=(10,8))

        # plot the loss
        plt.subplot(1, 2 ,1)
        plt.plot(epochs,train_loss,label="Train Loss")
        plt.plot(epochs,test_loss,label="Test Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        # plot the accucary
        plt.subplot(1, 2, 2)
        plt.plot(epochs,train_acc,label="Train Acc")
        plt.plot(epochs,test_acc,label="Test Acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

    # plot_loss_curves(model0_result)
    # plt.show()

    #

    Save("models","model0.pth",model=model0)
    save_results_txt("results","model0_results.txt",model0_result)













if __name__ == "__main__":
    main()