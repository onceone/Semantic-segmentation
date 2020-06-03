#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/4/24 18:20
# @Author : v
# @File : train.py
# @Software: PyCharm

import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import models

CLASS_NUMBERS = 2  # 分几类，这里分两类
HEIGHT = 416  # 图片的长
WIDHT = 416  # 图片的宽
batch_size = 4  # 一次处理的图片数


def customied_loss(y_true, y_pred):
    """ 自定义损失函数"""
    loss = binary_crossentropy(y_true, y_pred)
    return loss


def get_model():
    """ 获取模型，并加载官方预训练的模型参数 """

    # 获取模型
    model = models.main()
    # model.summary() cache_subdir='models_dir'

    # 下载模型参数
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    filename = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # 下载后保存的文件名
    checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    weights_path = get_file(filename, WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    # print(weights_path)

    # 加载参数
    model.load_weights(weights_path, by_name=True)

    # 编译
    model.compile(loss=customied_loss, optimizer=Adam(1e-3), metrics=['accuracy'])

    return model


def get_data():
    """
    获取获取样本和标签对应的行；获取训练集和验证集的数量
    :return: lines: 样本和标签的对应行；[num_train,num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines => ['1.jpg;1.png\n', '10.jpg;10.png\n',.....]
    # .jpg:样本 ； .png: 标签
    with open('./dataset/trainData/train.txt', 'r') as f:
        lines = f.readlines()
    # print(lines)

    # 打乱行，打乱数据有利于训练
    np.random.seed(10101)  # 设置随机种子，
    np.random.shuffle(lines)
    np.random.seed(None)

    # 切分训练样本，90% 训练；10% 验证
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    # print(num_val)

    return lines, num_train, num_val


def set_callbacks():
    """ 设置回调函数"""

    # 1. 有关回调函数的设置（callbacks)
    logdir = os.path.join("callbacks")
    print(logdir)
    if not os.path.exists(logdir):  # 如果没有文件夹
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        ModelCheckpoint(output_model_file, save_best_only=True, save_freq='epoch'),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-3, patience=10)
    ]

    return callbacks, logdir


def generate_arrays_from_file(lines, batch_size):
    """
    生成器，读取图片，并对图片处理，生成（样本，标签）
    :param lines: 样本和标签的对应行 ：（119.jpg;119.png)=(样本，标签）
    :param batch_size: 一次处理的图片数
    :return:  返回 （样本，标签）
    """

    numbers = len(lines)  # 传进来样本数据的总长度

    read_line = 0  # 读取的行，是否读取完一个周期（所有）数据
    while True:

        x_train = []  # 样本
        y_train = []  # 标签

        # 一次获取batch_size大小的数据

        for t in range(batch_size):

            # print(t)
            # 如果读取完一个周期数据，后，将数据打乱,重新开始
            if read_line == 0:
                np.random.shuffle((lines))

            # 处理训练样本 dataset/jpg ：训练图片；dataset/png：训练图片的标签

            # 1. 获取训练文件的名字，名字如，42.jpg
            train_x_name = lines[read_line].split(';')[0]

            # 根据图片名字读取图片
            img = Image.open('./dataset/trainData/jpg' + '/' + train_x_name)
            # img.show()
            # print(img.size)
            img = img.resize((WIDHT, HEIGHT))  # 改变图片的大小->(416,416)
            img_array = np.array(img)  # 图片转换成数组
            # print('训练数据shape:', img_array.shape)

            img_array = img_array / 255  # 标准化 img_array.shape=（416，416，3）

            x_train.append(img_array)  # 添加到训练样本

            # 2. 获取训练样本标签的名字,名字如：42.png；标签是一张图片
            train_y_name = lines[read_line].split(";")[1].replace('\n', '')

            # 根据图片名字读取图片
            img = Image.open('./dataset/trainData/png' + '/' + train_y_name)
            # img.show()
            # print(train_y_name)
            img = img.resize((int(WIDHT / 2), int(HEIGHT / 2)))  # 改变图片大小 -> (208,208)。标签（图片）大小208*208
            img_array = np.array(img)  # 图片转换成数组 img_array.shape=(208,208,3)
            # img_array,三个通道数相同，没法做交叉熵计算，所以要进行下面“图层分层”

            # 生成标签，标签的shape是(208,208,class_numbers)=(208,208,2),里面的值全为0
            labels = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
            # print('label shape: ', labels.shape)

            # 下面将(208,208,3) => (208,208,2),不仅是通道数的变化，还有，
            # 原本背景和斑马线在一个通道里面，现在将斑马线和背景放在不同的通道里面。
            # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
            # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
            # 相当于合并的图层分层！！！！
            for cn in range(CLASS_NUMBERS):  # range(0,2) => 0,1
                # 标签数组中，斑马线的值为1，其他为0 （三个通道值相同！！！），所以下面选第0通道
                labels[:, :, cn] = (img_array[:, :, 0] == cn).astype(int)
            labels = np.reshape(labels, (-1, CLASS_NUMBERS))  # (208,208,2)=>(208*208,2)
            y_train.append(labels)

            # 遍历所有数据，记录现在所处的行，读取完所有数据后，read_line=0,上面会打乱，重新开始
            read_line = (read_line + 1) % numbers

        yield (np.array(x_train), np.array(y_train))


def main():
    # 获取已建立的模型，并加载官方与训练参数，模型编译
    model = get_model()
    # 打印模型摘要
    # model.summary()

    # 获取样本（训练集&验证集） 和标签的对应关系，trian_num,val_num
    lines, train_nums, val_nums = get_data()

    # 设置回调函数 并返回保存的路径
    callbacks, logdir = set_callbacks()

    # 生成样本和标签
    generate_arrays_from_file(lines, batch_size=4)

    # 训练
    model.fit_generator(generate_arrays_from_file(lines[:train_nums], batch_size),
                        steps_per_epoch=max(1, train_nums // batch_size),
                        epochs=50, callbacks=callbacks,
                        validation_data=generate_arrays_from_file(lines[train_nums:], batch_size),
                        validation_steps=max(1, val_nums // batch_size),
                        initial_epoch=0)

    save_weight_path = os.path.join(logdir,'last.h5') # 保存模型参数的路径

    model.save_weights(save_weight_path)


if __name__ == '__main__':
    main()
