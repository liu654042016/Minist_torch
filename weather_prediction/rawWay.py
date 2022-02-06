import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler
import torch

#数据读取
path="D:\\study\\deeplearing\\Minist_torch\\data\\Weather\\NewTemp\\tempsssssss\\temps1.csv"
data = pd.read_csv(path)

#数据维度
#print(data.head())

#print(data.shape)

#print(type(data))

#数据预处理
dates = pd.PeriodIndex(year = data["year"], month = data["month"], day = data["day"], freq = "D").astype(str)
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
#print(dates[:5])

#编码格式转换
data = pd.get_dummies(data)
#print(data.head())

#画图
plt.style.use("fivethirtyeight")
register_matplotlib_converters()

#标签
labels = np.array(data["actual"])

#取消标签
data = data.drop(["actual"], axis=1)
#print(data.head())

#保存一下列名
feature_list = list(data.columns)

#格式转换
data_new = np.array(data)

data_new = StandardScaler().fit_transform(data_new[:,:14])
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",data_new.shape)
#print(data_new)


####################构建网络模型
x = torch.tensor(data_new)
y = torch.tensor(labels)

#权重参数初始化
weights1 = torch.randn((14, 128), dtype=float, requires_grad=True)
biases1 = torch.randn(128, dtype=float, requires_grad=True)
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
biases2 = torch.randn((1), dtype=float, requires_grad=True)


learning_rate = 0.001
losses = []

for i in range(1000):
    #计算隐层
    hidden = x.mm(weights1) + biases1
    #加入激活函数
    hidden = torch.relu(hidden)
    #预测结果
    predictions = hidden.mm(weights2) + biases2
    #计算损失
    loss = torch.mean((predictions-y)**2)

    #打印损失
    if i%100 == 0:
        print("loss:", loss)
    #反向传播计算
    loss.backward()

    #更新参数
    weights1.data.add_(-learning_rate*weights1.grad.data)
    biases1.data.add_(-learning_rate*biases1.grad.data)
    weights2.data.add_(-learning_rate*weights2.grad.data)
    biases2.data.add_(-learning_rate*biases2.grad.data)

    #每次迭代清空
    weights1.grad.data.zero_()
    biases1.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()



#数据可视化
def graph1():
    #创建子图
    f, ax = plt.subplots(2, 2, figsize=(10, 10))

    #标签值
    ax[0, 0].plot(dates, labels, color="#ADD8E6")
    ax[0, 0].set_xticks([""])
    ax[0, 0].set_ylabel("Temperature")
    ax[0, 0].set_title("Max Temp")

    #昨天
    ax[0, 1].plot(dates, data["temp_1"], color="#87CEFA")
    ax[0, 1].set_xticks([""])
    ax[0, 1].set_ylabel(["Temperature"])
    ax[0, 1].set_title("Previous Max Temp")


    #前天
    ax[1, 0].plot(dates, data["temp_2"], color="#00BFFF")
    ax[1, 0].set_xticks([""])
    ax[1, 0].set_xlabel("Date")
    ax[1, 0].set_ylabel("Temperature")
    ax[1, 0].set_title("two days prior max temp")

    #朋友
    ax[1, 1].plot(dates, data["friend"], color="#1E90FF")
    ax[1, 1].set_xticks([""])
    ax[1, 1].set_xlabel("Date")
    ax[1, 1].set_ylabel("Temperature")
    ax[1, 1].set_title("Friend Estimate")

    plt.show()

if __name__ == "__main__":
    graph1()


