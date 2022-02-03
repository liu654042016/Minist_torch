import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
import warnings
import torch.nn.functional as F


class TemptureModel(torch.nn.Module):
    def __init__(self, input_shape) -> None:
        super(TemptureModel, self).__init__()
        print(".........", input_shape)
        self.input_size = input_shape
        self.LinearInput = torch.nn.Linear(self.input_size, 128)
        self.LinearOutput = torch.nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.sigmoid(self.LinearInput(x))
        x = self.LinearOutput(x)
        return x


