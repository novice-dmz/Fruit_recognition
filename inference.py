import torch
import os
import models
import utils
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor()])
classes_mapper = utils.read_dictionary('output/classes.txt')
keys = list(classes_mapper.keys())


def image_deal(image):
    image = image.resize((100, 100))
    plt.imshow(image)
    plt.show()
    image = transform(image)
    image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
    return image


def get_state_dict(path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model


def get_inference_class_name(path):
    img = Image.open(path)
    img = image_deal(img)
    outputs = model(img)
    prediction = F.softmax(outputs, dim=1).detach().numpy()[0]
    index = np.argmax(prediction)
    class_name = keys[index]
    return class_name


if __name__ == '__main__':
    model = models.FruitsModel()
    model = get_state_dict('output/fruitsnet.pt')
    test_folder_path = 'input/test/'
    for filename in os.listdir(test_folder_path):
        img_path = (os.path.join(test_folder_path, filename))
        print(img_path, get_inference_class_name(img_path))



