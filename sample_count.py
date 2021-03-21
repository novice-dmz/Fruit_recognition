import os
import pandas as pd

"""
    从数据集文件夹中统计样本（与训练无关）
"""
train_dir = 'input/fruits-360/Training/'
test_dir = 'input/fruits-360/Test/'

# 从训练和测试目录创建数据框
train_df = pd.DataFrame(columns=['img_path', 'class'])
test_df = pd.DataFrame(columns=['img_path', 'class'])

i = 0
j = 0


# 遍历训练文件
for className in os.listdir(train_dir):
    if i == 1:
        break
    for filename in os.listdir(os.path.join(train_dir, className)):
        img_path = (os.path.join(train_dir, className, filename))
        train_df = train_df.append({'img_path': img_path, 'class': className}, ignore_index=True)
    i = i + 1

# 遍历测试文件
for className in os.listdir(test_dir):
    if j == 1:
        break
    for filename in os.listdir(os.path.join(test_dir, className)):
        img_path = (os.path.join(test_dir, className, filename))
        test_df = test_df.append({'img_path': img_path, 'class': className}, ignore_index=True)
    j = j + 1

# tf1 = train_df['class'].value_counts().to_frame().reset_index()
# tf1.rename(columns={"index": "class", "class": "train_counts"}, inplace=True)
# tf2 = test_df['class'].value_counts().to_frame().reset_index()
# tf2.rename(columns={"index": "class", "class": "test_counts"}, inplace=True)
# df = pd.merge(tf1, tf2, on='class')

v1 = train_df[0]
v2 = train_df[6]

