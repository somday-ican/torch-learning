import numpy as np
import matplotlib.pyplot as plt
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
w=1.0
def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)**2
def gradient(x,y):
    return 2*x*(w*x-y)
print('Predict before training',4,forward(4))
ep_list=[]
cost_list=[]
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_val=loss(x,y)
        grad_val=gradient(x,y)
        w-=grad_val*0.01
        print('\tgrand',grad_val,x,y)
    print('Epoch:',epoch,'w=',w,'loss',loss_val)
    print('Predict after training',4,forward(4))
    ep_list.append(epoch)
    cost_list.append(loss_val)
# a=int(input("请输入x的值："))
# print('x=',a,'y=',forward(a))
plt.title('Cost in each epoch')
plt.plot(ep_list,cost_list)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()