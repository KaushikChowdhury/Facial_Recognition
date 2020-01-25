import os
import glob
import numpy as np
import pandas as pd
import argparse
import json

import torch
import cv2
from torch.utils.data import Dataset,DataLoader
# from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
import transformations
from torchvision.transforms import Compose

str2bool = lambda x : (str(x).lower == "true")

parser = argparse.ArgumentParser(description = "Facial Keypoint detection")

parser.add_argument("--config", "-c", help = "json config file", default = "./config_all.json")
parser.add_argument("--use_gpus", "-u", type = str2bool ,help = "flag to use gpus or not", default = False)

args = parser.parse_args()

#load config file
with open(args.config) as data_file:
    config = json.load(data_file)


class FacialKeypointDataset(Dataset):
    """
    Facial Landmark detection
    """
    def __init__(self,csv_keypoints_path, root_directory, transform = None):
        self.root = root_directory
        self.transform = transform
        self.keypoints_frame = pd.read_csv(csv_keypoints_path)
        
    def __len__(self):
        return len(self.keypoints_frame)

    def __getitem__(self, index):
        image_name = os.path.join(self.root, self.keypoints_frame.iloc[index,0])
        image =  cv2.imread(image_name)
        

        if (image.shape[2] == 4):
            image = image[:,:,0:3]

        keypoints = self.keypoints_frame.iloc[index, 1:].values
        keypoints = keypoints.astype("float").reshape(-1,2)  # creating two columns with x and y coordinates

        sample = {"image" : image, "keypoints": keypoints}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    csv_keypoints_path_train = config["csv_keypoints_path_train"]
    csv_keypoints_path_test = config["csv_keypoints_path_test"]
    root_directory = config["root_directory"]
    data_transform = Compose([transformations.Rescale(250), transformations.Normalize(), transformations.ToTensor()])
    
    
    batch_size = int(config["batch_size"])
    train_data = FacialKeypointDataset(csv_keypoints_path_train, root_directory, transform = data_transform)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers=1)

    test_data = FacialKeypointDataset(csv_keypoints_path_test, root_directory, transform = data_transform)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers=1)

    





    


