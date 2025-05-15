import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x

