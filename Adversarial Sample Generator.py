import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
from RPN import RelationModule

# 定义对抗扰动生成器模型
class AdversarialPerturbation(nn.Module):
    def __init__(self, input_shape):
        super(AdversarialPerturbation, self).__init__()
        self.input_shape = input_shape
        self.model = models.vit_small_patch16_224(pretrained=True)

    def forward(self, x):
        return self.model(x)


# 加载数据集和分类器
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
train_dataset = datasets.ImageFolder('/path/to/train/dataset', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


# 定义优化器和学习率
FeatureExtractor=RelationModule()
generator = AdversarialPerturbation(input_shape=(3, 224, 224)).to('cuda')
feature_extractor = FeatureExtractor().to('cuda')
optimizer = optim.Adam(generator.parameters(), lr=0.001)

# 定义欧式距离和相似度损失函数
def distance_loss(d1, d2, alpha):
    distance = torch.norm(d1 - d2, dim=1)
    similarity = torch.cosine_similarity(d1, d2, dim=1)
    loss = distance - alpha * similarity
    return loss.mean()

# 训练对抗扰动生成器
num_epochs = 10
alpha = 0.5
for epoch in range(num_epochs):
    for x, y in train_loader:
        x = x.to('cuda')
        y = y.to('cuda')
        generator.zero_grad()
        perturbation = generator(x)
        adversarial_image = x + perturbation
        d1 = feature_extractor(x)
        d2 = feature_extractor(adversarial_image)
        loss = distance_loss(d1, d2, alpha)
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# 使用训练好的对抗扰动生成器生成对抗性扰动
test_image = Image.open('/path/to/test/image.jpg')
test_image_tensor = transform(test_image).unsqueeze(0).to('cuda')
perturbation = generator(test_image_tensor)
adversarial_image = test_image_tensor + perturbation