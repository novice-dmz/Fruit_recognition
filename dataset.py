import os
import utils
from torch.utils.data import Dataset
from PIL import Image


# 根据目录路径构建数据集
def make_dataset(dirpath):
    images = []
    classes = []
    # 遍历文件
    for className in os.listdir(dirpath):
        classes.append(className)
        for filename in os.listdir(os.path.join(dirpath, className)):
            img_path = (os.path.join(dirpath, className, filename))
            item = (img_path, className)
            images.append(item)
    # 构建类名到索引的字典，以便于训练（例：'Apple Braeburn': 0, 'Apple Crimson Snow': 1）
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # 把字典写入文件classes.txt
    if os.path.exists('output/classes.txt') is False:
        utils.save_dictionary('output/classes.txt', class_to_idx)
    return images, class_to_idx


class FruitDataset(Dataset):
    def __init__(self, dirpath, transform=None):
        # images存放所有图像的地址和类元组， class_to_idx字典存放类名到索引的映射
        self.images, self.class_to_idx = make_dataset(dirpath)
        self.transform = transform  # 是否需要图像增强

    def __getitem__(self, index):
        img, target = self.images[index]
        img = Image.open(img).convert('RGB')
        # 将target英文映射成数字
        target = self.class_to_idx[target]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        # 返回图像的数量
        return len(self.images)
