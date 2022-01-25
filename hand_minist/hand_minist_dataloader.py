import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt


transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3018,))])
train_dataset = datasets.MNIST('data/', train=True,transform = transformation,download=True)
test_dataset = datasets.MNIST('data/', train=False, transform = transformation, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 8, shuffle = True)


simple_data = next(iter(train_loader))
def show_images(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    print(image.shape)
    plt.imshow(image, cmap='gray')


if __name__ == "__main__":
    show_images(simple_data[0][8])

    