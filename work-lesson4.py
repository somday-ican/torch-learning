import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0,4.0,5.0,6.0]
y_data = [2.0,4.0,6.0,8.0,10.0,12.0]
w1=torch.tensor([1.0])
w2=torch.tensor([2.0])
b=torch.tensor([3.0])
w1.requires_grad=True
w2.requires_grad=True
b.requires_grad=True
def forward(x):
    return x*x*w1+x*w2+b

def loss(x,y):
    pred_y=forward(x)
    return (pred_y-y)**2

for epoch in range(1000):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        #根据梯度下降算法更新权重
        w1.data=w1-0.0001*w1.grad.data
        w2.data=w2-0.0001*w2.grad.data
        b.data=b-0.0001*b.grad.data
        #清除之前计算过的梯度
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('Epoch:',epoch,'loss=',l.item(),)
print("after training",7,forward(7).item())