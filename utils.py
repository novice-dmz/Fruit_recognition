import matplotlib.pyplot as plt
import numpy as np

def save_dictionary(path, dictionary):
    fw = open(path, 'w+')
    fw.write(str(dictionary))  # 把字典转化为str
    fw.close()


def read_dictionary(path):
    fr = open(path, 'r+')
    dictionary = eval(fr.read())  # 读取的str转换为字典
    fr.close()
    return dictionary


def draw_acc_graph(path):
    acc_array = np.loadtxt(path)
    train_acc, valid_acc = acc_array.tolist()
    plt.figure(figsize=(20, 10))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.show()
