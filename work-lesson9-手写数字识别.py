import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

'''第一步，数据加载'''
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307,0.3081)
])

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transform,download=True)

#dataloader部分
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

"""第二步，定义训练模型"""
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(28*28, 512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)

    def forward(self, x):
        x=x.view(-1,28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)       #最后一层无需使用relu激活函数，因为官方文档中说明了计算交叉熵损失无需归一的数据

model = Net()

"""第三步，定义损失函数和优化器"""
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.5)

"""第四步，写训练周期和测试函数"""
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,targets = data
        optimizer.zero_grad()#梯度清零可以靠前的原因是：只有反向传播才会计算梯度并存储，故只要在反馈前清零就可以了
        #前馈、反馈、更新
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            """此处返回最的概率值，以及那个值的索引-即预测结果，而预测值我们不需要，故使用_,作为占位符"""
            total += labels.size(0)#加上这个批量的总数
            correct+=(predicted==labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

for epoch in range(10):
    train(epoch)
    test()
