import numpy as np
import torch


xy=np.loadtxt('diabetes_data.csv',delimiter=' ',dtype=np.float32)
x_data=torch.from_numpy(xy[:,:-1])#因为要去掉最后一列
y_data=torch.from_numpy(xy[:,[-1]])#只取最后一列但不能作为向量，所以要加方括号

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(9,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model=Model()

criterion=torch.nn.MSELoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=0.01)

for epoch in range(1000):
    #forward
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print('epoch:',epoch,'loss:',loss.item())

    #backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()

print("-" * 30)
print(f"预测值: {model(x_data[0:1]).item():.2f}")
print(f"真实值: {y_data[0].item():.2f}")
