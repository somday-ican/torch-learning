import torch
import torch.nn as nn

x_data=torch.Tensor([[1.0],[2.0],[3.0],[4.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0],[8.0]])

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
  y_pred = model(x_data)
  loss = criterion(y_pred, y_data)
  print('epoch:',epoch,'loss:',loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test=torch.Tensor([[5.0]])
y_test=model(x_test)
print('y_test=',y_test.item())#这里需要使用item()，否则打出来会是一个张量

