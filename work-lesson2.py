import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 修改后的前向传播函数
def forward(x, w, b):
    return x * w + b

# 修改后的损失函数
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) * (y_pred - y)

# 初始化存储列表
w_list = []
b_list = []
mes_list = []

# 双层循环遍历 w 和 b
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, w, b)
            l_sum += loss_val
        # 计算并存储 MSE
        w_list.append(w)
        b_list.append(b)
        mes_list.append(l_sum / 3)

# 3D 可视化
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w_list, b_list, mes_list, c=mes_list, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('loss')
plt.title('Loss Function Landscape for y = x*w + b')
plt.show()