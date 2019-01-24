import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.relu = nn.LeakyReLU()

        self.conv1 = weight_norm(nn.Conv1d(1025, 1024, 3, 1, 1))

        self.conv2 = weight_norm(nn.Conv1d(1024, 512, 3, 2, 1))

        self.conv3 = weight_norm(nn.Conv1d(512, 256, 3, 2, 1))

    def forward(self, x, level=0):
        osize = x.size()

        x = self.relu(self.conv1(x))
        x1 = x

        x = self.relu(self.conv2(x))
        x2 = x

        x = self.relu(self.conv3(x))

        return osize, x1, x2, x

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()

        self.relu = nn.LeakyReLU()
        self.Sigmoid = nn.Sigmoid()

        self.conv1 = weight_norm(nn.Conv1d(256, 128, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(128, 64, 3, 1, 1))
        self.conv3 = weight_norm(nn.Conv1d(64, 32, 3, 1, 1))
        self.linear1 = weight_norm(nn.Linear(32*13, 256))
        self.linear2 = weight_norm(nn.Linear(256, 256))

        
    def forward(self, x):
        bs = x.size(0)

        x = F.max_pool1d(self.relu(self.conv1(x)), 2)
        x = F.max_pool1d(self.relu(self.conv2(x)), 2)
        x = F.max_pool1d(self.relu(self.conv3(x)), 2)

        x = x.view(bs, -1)

        x = self.relu(self.linear1(x))
        x = self.Sigmoid(self.linear2(x))

        return x

class NetC(nn.Module):
    def __init__(self, num_class):
        super(NetC, self).__init__()

        self.num_class = num_class
        
        self.softmax = nn.Softmax(dim=1)
        self.linear3 = weight_norm(nn.Linear(256, num_class))
        
    def forward(self, x):

        x = self.softmax(self.linear3(x))

        return x


