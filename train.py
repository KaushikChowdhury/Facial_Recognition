from data_loader import FacialKeypointDataset
import model

import os
import glob
import numpy as np
import pandas as pd
import argparse
import json

import cv2
from torch.utils.data import Dataset,DataLoader
# from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
import transformations
from torchvision.transforms import Compose
from model import Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


str2bool = lambda x: (str(x).lower() == 'true')

parser = argparse.ArgumentParser(description = "Facial Keypoint detection")

parser.add_argument("--config", "-c", help = "json config file", default = "./config_all.json")
parser.add_argument("--use_gpus", "-u", type = str2bool ,help = "flag to use gpus or not", default = True)

args = parser.parse_args()

#load config file
with open(args.config) as data_file:
    config = json.load(data_file)

device = torch.device("cuda" if (args.use_gpus and torch.cuda.is_available()) else "cpu")
best_prec1 = 0



if args.use_gpus:
        net = Net().to(device)
else:
    net = Net()

if __name__ == "__main__":
    csv_keypoints_path_train = config["csv_keypoints_path_train"]
    csv_keypoints_path_test = config["csv_keypoints_path_test"]
    root_directory = config["root_directory"]
    data_transform = Compose([transformations.Rescale(250), transformations.RandomCrop(224), transformations.Normalize(), transformations.ToTensor()])


    batch_size = int(config["batch_size"])
    train_data = FacialKeypointDataset(csv_keypoints_path_train, root_directory, transform = data_transform)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers=1)

    test_data = FacialKeypointDataset(csv_keypoints_path_test, root_directory, transform = data_transform)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers=1)

    # test_images, test_outputs, gt_pts = net_sample_output()

    # # print out the dimensions of the data to see if they make sense
    # print(test_images.data.size())
    # print(test_outputs.data.size())
    # print(gt_pts.size())

    

    #preparing for train
    criterion = nn.SmoothL1Loss()
# criterion = nn.MSELoss() # not giving much better accuracy as of the all attempts I made.

    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    net.train()
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):

        running_loss = 0

        for batch_i, data in enumerate(train_loader):
            images = data["image"]
            keypoints = data["keypoints"]

            # flatten pts
            keypoints = keypoints.view(keypoints.size(0), -1)

            # convert variables to floats for regression loss
            keypoints = keypoints.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, keypoints)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            loss.backward()

            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0


    
    


    




