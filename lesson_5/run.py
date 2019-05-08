# encoding:utf-8
# 预处理数据
from __future__ import print_function, division
from scipy.io import loadmat as load
import matplotlib.pyplot as plt
import numpy as np


def reformat(samples, labels):
    # 改变原始数据的形状
    #       0          1           2            3        ->        3           0           1           2
    # (图片高,图片宽,通道数, 图片数)  -> (图片数, 图片高,通道数,通道数)
    new = np.transpose(samples, (3, 0, 1, 2))
    # 0 用 10 表示
    # 0 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 2 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    labels = np.array([x[0] for x in labels])
    one_hot_labels = []
    for num in labels:
        one_hot = [0.0] * 10
        if num == 10:
            one_hot[0] = 1.0
        else:
            one_hot[num] = 1.0
        one_hot_labels.append(one_hot)
    labels = np.array(one_hot_labels).astype(np.float32)
    return new, labels


def normalize(samples):
    # 灰度化图片 三色通道 -> 单色通道 节省内存,也可以加快训练速度
    # （R + G + B）/  3
    #  将图片的色彩值 0 ~ 255 线性映射到 -1.0 ~ 1.0
    a = np.add.reduce(samples, keepdims=True, axis=3)  # shape (图片数，图片高，图片宽，通道数)
    a = a / 3.0
    return a / 128.0 - 1.0


def distribution(labels, name):
    # 查看数据分布
    count = {}
    for label in labels:
        key = 0 if label[0] == 10 else label[0]
        if key in count:
            count[key] += 1
        else:
            count[key] = 1
    x = []
    y = []
    for k, v in count.items():
        x.append(k)
        y.append(v)
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center', alpha=0.5)
    plt.xticks(y_pos, x)
    plt.ylabel('Count')
    plt.title(name + ' Label Distribution')
    plt.show()


def inspect(samples, labels, i):
    # 查看一张图片
    # if samples.shape[3] == 1:
    #     shape = samples.shape
    #     samples = samples.reshape(shape[0], shape[1], shape[2])
    print(labels[i])
    plt.imshow(samples[i])
    plt.show()


train_data = load('../data/train_32x32.mat')
test_data = load('../data/test_32x32.mat')

print('Train Data Samples Shape: ', train_data['X'].shape)
print('Train Data  Labels Shape: ', train_data['y'].shape)

print('Test Data  Samples Shape: ', test_data['X'].shape)
print('Test Data   Labels Shape: ', test_data['y'].shape)

train_samples = train_data['X']
train_labels = train_data['y']

n_train_samples, n_train_labels = reformat(train_samples, train_labels)

# 不能运行
# _train_samples = normalize(n_train_samples)
# _train_labels = normalize(n_train_labels)

# inspect(n_train_samples, n_train_labels, 123)
distribution(train_labels, "Train")
distribution(test_data['y'], "Test")