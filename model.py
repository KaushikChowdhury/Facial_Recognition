import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #define all the layers of this network. 
        #1. This network takes in a square( same height and width), grayscale image as input 224 * 224
        #2. (W - F)/S  + 1  = (224 - 5) + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5) #  input = (1, 224, 224); output = (32, 220, 220)

        self.fc1 = torch.nn.Linear(32*220*220, 136)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x 
