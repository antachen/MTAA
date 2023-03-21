import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BeitFeatureExtractor, BeitModel
import requests


class BEiTEncoder():
    def __init__(self, model_name='microsoft/beit-base-patch16-224-pt22k'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = BeitFeatureExtractor.from_pretrained(model_name)
        self.model = BeitModel.from_pretrained(model_name).to(self.device)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        image = Image.open(image_path)
        image = self.transforms(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image).last_hidden_state
            features = features[:, 0, :] # 取CLS Token作为图像的特征向量
        features = features.cpu().numpy()
        return features

Encoder = BEiTEncoder()
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
features = Encoder.extract_features(requests.get(url, stream=True).raw)
print(features.shape)
