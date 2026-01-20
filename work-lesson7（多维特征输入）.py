import numpy as np
import torch


xy=np.loadtxt('diabetes_data.csv',delimiter=' ',dtype=np.float32)
x_data=torch.from_numpy(xy[:,:-1])#因为要去掉最后一列
y_data=torch.from_numpy(xy[:,[-1]])#只取最后一列但不能作为向量，所以要加方括号

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x=self.linear1(x)
        x=self.sigmoid(x)
        x=self.linear2(x)
        x=self.sigmoid(x)
        x=self.linear3(x)
        x=self.sigmoid(x)
        return x

model=LogisticRegression()

criterion=torch.nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(100):
    #forward
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print('epoch:',epoch,'loss:',loss)

    #backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()

