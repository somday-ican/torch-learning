import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 1. 数据预处理与自定义 Dataset
def load_otto_data():
    # 加载数据
    train_df = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/train.csv')
    test_df = pd.read_csv('/kaggle/input/otto-group-product-classification-challenge/test.csv')

    # 提取特征（从 feat_1 到 feat_93）
    features = [f'feat_{i}' for i in range(1, 94)]
    x_train = train_df[features].values.astype('float32')
    x_test = test_df[features].values.astype('float32')
    test_ids = test_df['id'].values

    # 标签处理：将 "Class_1" 等转为 0-8 的整数
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df['target'])

    # 标准化（对神经网络至关重要）
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, test_ids, encoder


x_train_np, y_train_np, x_test_np, test_ids, encoder = load_otto_data()


class OttoDataset(Dataset):
    def __init__(self, x, y=None):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y) if y is not None else None

    def __getitem__(self, index):
        if self.y_data is not None:
            return self.x_data[index], self.y_data[index]
        return self.x_data[index]

    def __len__(self):
        return self.x_data.shape[0]


# 准备 DataLoader
batch_size = 64
train_dataset = OttoDataset(x_train_np, y_train_np)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 2. 定义网络 (输出改为 9)
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(93, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 9)  # 修改：OTTO 是 9 分类

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()

# 3. 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 4. 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.long())  # 交叉熵要求标签为 Long 类型
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 100))
            running_loss = 0.0


# 5. 生成提交文件的预测函数
def generate_submission():
    model.eval()
    test_tensor = torch.from_numpy(x_test_np)

    with torch.no_grad():
        # 1. 得到原始输出（Logits）
        outputs = model(test_tensor)
        # 2. 关键：Kaggle 需要概率，使用 Softmax 转换
        #
        probabilities = F.softmax(outputs, dim=1).numpy()

    # 3. 构建结果 Dataframe
    # 动态获取 Class_1, Class_2... 的列名
    columns = encoder.classes_
    submission = pd.DataFrame(probabilities, columns=columns)
    submission.insert(0, 'id', test_ids)

    submission.to_csv('submission.csv', index=False)
    print("Submission file saved!")


# --- 执行 ---
for epoch in range(60):  # 运行 60 轮示例
    train(epoch)

generate_submission()