import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

class uncertaintyNet(nn.Module):
    def __init__(self):
        super(uncertaintyNet, self).__init__()
        
        
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 5) 
        #print("Liang XU")
        self.outactiv = torch.nn.Tanh(); 
        self.device = "cpu"
        self.out = None
        
        
    def forward(self, 
                distance, 
                angle,
                n_points):
        
        vec = np.zeros([1, 3])
        
        vec[0][0] = distance
        vec[0][1] = angle
        vec[0][2] = n_points
        torch_data_vec = torch.from_numpy(vec)
        vecInput = torch_data_vec.to(self.device, dtype=torch.float)
        
        x = self.fc1(vecInput)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        x = 0.1 * self.outactiv(x)
        self.out = x
        return x
