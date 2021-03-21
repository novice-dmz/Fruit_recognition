import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataset import FruitDataset
from torchvision import transforms


class FruitsModel(nn.Module):
    def __init__(self):
        super(FruitsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 131)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # 调试模型
    model = FruitsModel()
    img = Image.open('input/fruits-360/Training/Salak/0_100.jpg').convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = img.resize_(1, 3, 100, 100)
    outputs = model(img)
    # probs = F.softmax(outputs, dim=1).detach().numpy()[0]
    # pred = np.argmax(args3)
    args1 = F.softmax(outputs, dim=1)
    args2 = args1.detach()
    args3 = args2.numpy()[0]
    pred = np.argmax(args3)
    print(pred)
    classes_mapper = FruitDataset('input/fruits-360/Training/').get_classes_mapper()
    keys = list(classes_mapper.keys())
    print(keys[pred])
