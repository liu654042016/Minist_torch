#https://www.cnblogs.com/sunshine-66/p/15516437.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing

def test():
    print("aaaaaaaaaaa")

def get_data(path):
    if not path: return
    features  = pd.read_csv(path)
def format_data(data):
    features = pd.get_dummies(features)
    features.head(5)
    
    labels = np.array(features['actual'])
    features = features.drop('actual', axis=1)

    feature_list = list(features.columns)

    features = np.array(features)
    input_features = preprocessing.StandardScaler().fit_transform(features)

    return input_features


class DataReader():
    def __init__(self) -> None:
        self.features = None
    def __init__(self, path) -> None:
        self.path  = path
        self.feature_list = None
        self.labels = None
        if  not self.path:
            self.features = None
        else:
            self.features = self._get_data(self.path)
            #self.features = self.format_data()
    def get_format_data(self):
        self.features = pd.get_dummies(self.features)
        self.features.head(5)

        labels = np.array(self.features['actual'])
        self.labels = labels
        self.feature_list = list(self.features.columns)
        self.features = np.array(self.features)
        
        input_features = preprocessing.StandardScaler().fit_transform(self.features)
        print(len(input_features))
        return input_features, labels
    def get_labels(self):
        pass


    
    def _get_data(self, path):
        print(path)
        if not path: return None
         
        features = pd.read_csv(path)
        print(features.shape)
        return features
    
        


if __name__ == '__main__':
    path="D:\\study\\deeplearing\\Minist_torch\\data\\Weather\\NewTemp\\tempsssssss\\temps1.csv"
    my = DataReader(path)
    my.get_format_data()





