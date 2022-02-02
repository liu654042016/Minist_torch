import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from model import TemptureModel
from data import DataReader






def lossCalculator(predictions, labels):
    return coss(predictions , labels)



def train(input_features, labels,batch_size, optimizer):
    losses = []  
    for i in range(1000):
        batch_loss = []
        for start in range(0, len(input_features), batch_size):
            end = start + batch_size if start+batch_size < len(input_features) else len(input_features)
            xx = torch.tensor(input_features[start : end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(labels[start : end], dtype= torch.float, requires_grad=True)
            prediction = TemptureModel(xx)
            loss = lossCalculator(prediction, yy)
            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            batch_loss.append(loss.data.numpy())

        if i % 100 ==0:
            losses.append(np.mean(batch_loss))
            print(i, np.mean(batch_loss))




if __name__ == '__main__':
    
    temptureModel = TemptureModel(input_features)
    coss = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(temptureModel.parameters(), lr = 0.001)
