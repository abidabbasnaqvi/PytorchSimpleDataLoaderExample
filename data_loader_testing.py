from __future__ import division
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets, transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import  glob
import  os
import  shutil

from torch.utils.data import Dataset
from torch.utils.data import DataLoader



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
# BY: NAQVI, SYED ABID ABBAS
# PHD STUDENT: at SHANGHAI JIAO-TONG UNIVERSITY from Pakistan
# This example code show to design a DataLoader for Pytorch.
#-----------------------------------------------------------------------------------------------------------------------
# The basic concept is taken from  Vishu Subramanian book Deep Learning with Pytorch.
# Data can be downloaded from the link below
# https://www.kaggle.com/c/dogs-vs-cats/data
# Dataset contains 25,000 image of dogs and cats.
#-----------------------------------------------------------------------------------------------------------------------
# Download the dataset from the given link by default the name of the zip file is train.zip.
# What you have do is create a directory, then place train.zip inside that and extract it, and creat one more directory
# name 'valid'.
# Basically, what code it create two sub directories namely, dog and cat inside tain and valid directory.
# Then this program scans over all the images and places dogs in dog sub-directory and cats in cat sub-directory  of
# train and valid directory.
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


directory_uri = '/home/thaqafi/PycharmProjects/Neural_Transfer/dogsncats/train/'
print("Pytorch Version:", torch.__version__)
dataset_directory = os.getcwd() + "/dogsncats/"
print("The project current working directory-: ", os.getcwd())
def DogsAndCatsDataset(Dataset):

    def __init__(self, root_dir , size=(244,244), transform = None):
        self.size = size
        self.files =root_dir
        self.transform = transform


    def __len__(self):
        return(self.files)
    
    def __getitem_(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img, label

image_files = os.path.join(directory_uri, '*/*.jpg')

# Counting total number of files in downloaded DataSet.

paths, directories_in_cwd, files_in_cwd = next(os.walk(directory_uri))
no_of_images =  len(files_in_cwd)

#os.mkdir(os.path.join(dataset_directory, 'valid'), mode=511 )
print("Validation Set directory is being created -----")

print("The total number of images is :",  no_of_images)
shuffle = np.random.permutation(no_of_images)


print(files_in_cwd[0])
## Creating two seperate folder for cats and dogs in train and  valid folder

for  top_dirs in ['train', 'valid']:
    for sub_dirs in ['dog', 'cat']:
        if not os.path.exists(os.path.join(dataset_directory, top_dirs, sub_dirs)):
            os.mkdir(os.path.join(dataset_directory, top_dirs, sub_dirs))
            print('Folder created: ', os.path.join(dataset_directory, top_dirs, sub_dirs))

print("#################################################")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


####  Copying a subset of images into  valid folder
###   Copying is such that, dog goes into dog-folder, cat into cat-folder.

for i in shuffle[:2000]:
    print(i)
    print(files_in_cwd[i])
    folder = files_in_cwd[i].split('/')[-1].split('.')[0]
    print('Folder: ', folder)
    print("################################")

    image = files_in_cwd[i].split('/')[-1]
    if not os.path.exists(os.path.join(dataset_directory, 'valid', folder, image)):
        print(os.path.join(dataset_directory, 'valid', folder, image))
        shutil.copy(directory_uri + files_in_cwd[i], os.path.join(dataset_directory,'valid',folder), follow_symlinks=True )

####  Copying a subset of images into  training folder
###   Copying is such that, dog goes into dog-folder, cat into cat-folder.
for i in shuffle[2000:]:
    folder = files_in_cwd[i].split('/')[-1].split('.')[0]
    image = files_in_cwd[i].split('/')[-1]
    if not os.path.exists(os.path.join(dataset_directory, 'train', folder, image)):
        shutil.copy(directory_uri + files_in_cwd[i], os.path.join(dataset_directory, 'train', folder))



################## TRANSFORMING THE IMAGE #######################33

simple_transform =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

train = datasets.ImageFolder(dataset_directory + "train/", simple_transform)
valid = datasets.ImageFolder(dataset_directory + "valid/", simple_transform)

train_dataLoader  = torch.utils.data.DataLoader(train, batch_size= 64, num_workers=2)
valid_dataLaoder = torch.utils.data.DataLoader(valid, batch_size=64, num_workers=2)
print("Training dataset we just built : ", train.class_to_idx)
print("Valid dataset we just bulit : ", valid.class_to_idx)
print("#################################################")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
