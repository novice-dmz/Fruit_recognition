import torch
import numpy as np
import utils
from models import FruitsModel
from torchvision import models
from torchvision import transforms
from dataset import FruitDataset
from torch.utils.data import DataLoader

epochs = 25
batch_size = 128
learning_rate = 0.001

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
valid_loss_min = np.Inf

# 使用transforms转换函数，可以把PIL读取的图像转换成tensor
transform = transforms.Compose([transforms.ToTensor()])

train_set = FruitDataset('input/fruits-360/Training/', transform)
test_set = FruitDataset('input/fruits-360/Test/', transform)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
train_total_step = len(train_loader)
test_total_step = len(test_loader)

# 使用GPU计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FruitsModel()
# model = models.densenet121(pretrained=True)
# model.classifier = torch.nn.Linear(1024, 131)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(epoch):
    print(f'Epoch {epoch}\n')
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        if batch_idx % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                  .format(epoch, epochs, batch_idx, train_total_step, loss.item()))
    train_acc.append(100 * correct / total)
    train_loss.append(running_loss / train_total_step)
    print(f'\ntrain loss: {np.mean(train_loss):.8f}, train accuracy: {(100 * correct/total):.8f}%')


def test():
    batch_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, target)
            batch_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        valid_acc.append(100 * correct / total)
        valid_loss.append(batch_loss / test_total_step)
        print(f'validation loss: {np.mean(valid_loss):.8f}, validation accuracy: {(100 * correct / total):.8f}%\n')
        # 如果本次模型的损失更小则更新模型
        global valid_loss_min
        network_learned = batch_loss < valid_loss_min
        if network_learned:
            valid_loss_min = batch_loss
            torch.save(model.state_dict(), 'output/fruitsnet.pt')
            print('The detection accuracy rate has been improved, and the model has been updated!')


def save_accuracy():
    acc_dic = {train_acc[i]: valid_acc[i] for i in range(len(train_acc))}
    utils.save_dictionary('output/accuracy.txt', acc_dic)


if __name__ == '__main__':
    # 训练一轮测试一轮
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
    # 保存所有轮次的训练和测试的正确率以便于绘图
    save_accuracy()
