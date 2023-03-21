import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Encoder import BEiTEncoder
import requests

class RelationModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(RelationModule, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        linear_out = self.linear(x)
        mlp_out = self.mlp(x)
        output = torch.cat((linear_out, mlp_out), dim=1)
        return output

# 定义特征提取器
encoder = BEiTEncoder()
encoder = nn.Sequential(*list(encoder.children())[:-1])

# 定义关系保持模块和损失函数
relation_module = RelationModule(in_dim=512, out_dim=64)
criterion = nn.SmoothL1Loss()

# 定义超参数
learning_rate = 0.001
batch_size = 64
deltak = 1.0
num_epochs=100

# 定义数据集
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
train_set = datasets.ImageFolder(root="path/to/imagenet/train", transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 定义优化器
optimizer = optim.Adam(relation_module.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (data, target) in enumerate(train_loader):
        # 将数据送入GPU
        data, target = data.cuda(), target.cuda()

        # 将每个批次的数据拆分成两个图像样本，并提取特征
        h1 = encoder.extract_features(requests.get(data[:,0,:,:,:], stream=True).raw)
        h2 = encoder.extract_features(requests.get(data[:,1,:,:,:], stream=True).raw)

        # 将特征送入关系保持模块，计算损失函数并反向传播
        optimizer.zero_grad()
        output = relation_module(torch.cat((h1, h2), dim=1))
        loss = criterion(output[:,0], output[:,1]) + deltak * torch.abs(output[:,0]-output[:,1]).mean()
        loss.backward()
        optimizer.step()

        # 输出训练信息
        if i % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))


# 保存模型
torch.save(relation_module.state_dict(), 'relation_module.pth')

# 加载模型
torch.save(relation_module.state_dict(), 'relation_module.pth')
relation_module.load_state_dict('relation_module.pth')