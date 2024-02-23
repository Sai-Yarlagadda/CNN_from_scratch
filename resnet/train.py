import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyResnet, init_weights_kaiming
import torchvision.transforms as transforms


def setup_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cam(net, inputs, labels, idx):

    """
    Calculate the CAM.

    [input]
    * net     : network
    * inputs  : input data
    * labels  : label data
    * idx     : the index of the chosen image in a minibatch, range: [0, batch_size-1]

    [output]
    * cam_img : CAM result
    * img     : raw image

    [hint]
    * Inputs and labels are in a minibatch form
    * You can choose one images from them for CAM by idx.
    """

    net.eval()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs, feat_map = net(inputs, return_embedding=True)
        
        ## Find the class with highest probability
        ## Obtain the weight related to that class
        # ----- TODO -----

        ## Calculate the CAM
        ## Hint: you can choose one of the image (idx) from the batch for the following process
        # ----- TODO -----
        cam = None
        cam = cam.detach().cpu().numpy()


        ## Normalize CAM 
        ## Hint: Just minmax norm and rescale every value between [0-1]
        ## You will want to resize the CAM result for a better visualization
        ## e.g., the size of the raw image.
        # ----- TODO -----
        cam_img = None


        ## Denormalize raw images
        ## Hint: reverse the transform we did before
        ## Change the image data type into uint8 for visualization
        # ----- TODO -----
        img = inputs[idx].permute(1,2,0).detach().cpu().numpy()
        img = None

        return cam_img, img


if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 8
    lr = 1e-4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Set model
    ## Set the device to Cuda if needed
    ## Initialize all the parameters
    # ----- TODO -----
    net = MyResnet()


    ## Create the criterion and optimizer
    # ----- TODO -----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = net.parameters(), lr=lr)
    
    ## Load dataset
    normalize_param = dict(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
        )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), 
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(**normalize_param,inplace=True)
        ])

    # ----- TODO -----
    trainset = torchvision.datasets.CIFAR10(root='handout/code/deliverable3-5',
                                            train=True,
                                            transform=train_transform,
                                            download=True)
    trainloader = DataLoader(dataset=trainset, 
                             batch_size=batch_size,
                             shuffle=True)
    valset = torchvision.datasets.CIFAR10(root='handout/code/deliverable3-5',
                                          train=False,
                                          transform=val_transform,
                                          download=True)
    valloader = DataLoader(valset, batch_size, False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"LOAD DATASET: TRAIN/VAL | {len(trainset)}/{len(valset)}")


    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation 
    ## process into 2 different functions for each epoch
    for epoch in range(num_epoch): 

        # ----- TODO -----
        total_loss = 0
        correct = 0
        total_vals = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = net(inputs)
            values, prediction_output = output.max(1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            correct_vals = torch.eq(prediction_output, labels)
            correct += torch.sum(correct_vals).item()
            total_vals+=labels.size(0)
            total_loss+=loss.item()
        
        val_loss = 0
        correct_val = 0
        total_vals_val = 0
        for i, data in enumerate(valloader):
            inputs, labels = data
            output = net(inputs)
            vals, preds = output.max(1)
            true = torch.eq(preds, labels)
            correct_val += torch.sum(true).item()
            total_vals_val+=labels.size(0)

        print(f'train accuracy = {correct/total_vals}')
        print(f'val accuracy = {correct_val/total_vals_val}')
        


    print('Finished Training')

    ## Fetch the test image for CAM
    dataiter = iter(valloader)
    inputs, labels = next(dataiter)
    cam_img, img = cam(net, inputs, labels, idx=0) # idx could be changed

    ## Visualization
    ## Plot the loss and acc curves
    # ----- TODO -----


    ## Plot the CAM resuls as well as raw images
    ## Hint: You will want to resize the CAM result.
    # ----- TODO -----
    

