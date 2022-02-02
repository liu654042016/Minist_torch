#https://blog.csdn.net/wudibaba21/article/details/106940125
import torch
from torch.nn import Linear, ReLU
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


class Mnist_Net(nn.Module):
    def __init__(self):
        super(Mnist_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)#320 是根据4*4*20得到
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#model = Mnist_Net()
#model = model.cuda()
#optimizer = optim.SGD(model.parameters, lr=0.01)


def fit(optimizer,epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'Validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, target) in enumerate (data_loader):
       # data, target = data.cuda(),   target.cuda()
        data, target = data,   target
        data, target = Variable(data, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()#重置梯度
        output = model(data)
        loss = F.nll_loss(output, target)
        running_loss += F.nll_loss(output, target, size_average=False).item()#计算总的损失值
        preds = output.data.max(dim = 1, keepdim = True)[1]#预测概率转化为数字
      #  running_correct += preds.eq(target.data.view_as(preds)).cpu.sum()
        running_correct += preds.eq(target.data.view_as(preds)).sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. *running_correct/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

def Mytrain(model,optimizer, train_loader, test_loader):
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in range(1, 40):
        epoch_loss, epoch_accuracy = fit(optimizer,epoch, model, train_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(optimizer,epoch, model, test_loader, phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

if __name__ =="__main__":
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3018,))])
    train_dataset = datasets.MNIST('data/', train=True,transform = transformation,download=True)
    test_dataset = datasets.MNIST('data/', train=False, transform = transformation, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 8, shuffle = True)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",len(train_loader))
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",len(test_loader))


    print("aaaa")            
    model = Mnist_Net()
      #  model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    Mytrain(model, optimizer, train_loader, test_loader)