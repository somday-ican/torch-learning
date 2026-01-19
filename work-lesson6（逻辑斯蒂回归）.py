import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"#不加这个代码会报错
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])

class   LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
model = LogisticRegression()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(4000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('epoch=',epoch,'loss=',loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x=np.linspace(0,10,200)
x_test=torch.Tensor(x).view(-1,1)
y_test=model(x_test)
y=y_test.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],color='r')
plt.xlabel('Hours')
plt.ylabel('P(Past)')
plt.show()

