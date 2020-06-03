#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/4/23 21:49
# @Author : v
# @File : models.py
# @Software: PyCharm
import os, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

class_number = 2  # 分类数

print('python版本为：' + sys.version)
print('tensorflow版本为：' + tf.__version__)


def encoder(input_height, input_width):
    """
    语义分割的第一部分，特征提取，主要用到VGG网络，函数式API
    :param input_height: 输入图像的长
    :param input_width: 输入图像的宽
    :return: 返回：输入图像，提取到的5个特征
    """

    # 输入
    img_input = Input(shape=(input_height, input_width, 3))

    # 三行为一个结构单元，size减半
    # 416,416,3 -> 208,208,64,
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f1 = x  # 暂存提取的特征

    # 208,208,64 -> 104,104,128
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f2 = x  # 暂存提取的特征

    # 104,104,128 -> 52,52,256
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f3 = x  # 暂存提取的特征

    # 52,52,256 -> 26,26,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f4 = x  # 暂存提取的特征

    # 26,26,512 -> 13,13,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f5 = x  # 暂存提取的特征

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=416, input_width=416, encoder_level=3):
    """
    语义分割的后半部分，上采样，将图片放大，
    :param feature_map_list: 特征图（多个），encoder得到
    :param class_number: 分类数
    :param input_height: 输入图像长
    :param input_width: 输入图像宽
    :param encoder_level: 利用的特征图，这里利用f4
    :return: output , 返回放大后的特征图 （208*208,2）
    """
    # 获取一个特征图，特征图来源encoder里面的f1,f2,f3,f4,f5; 这里获取f4
    feature_map = feature_map_list[encoder_level]

    # 解码过程 ，以下 （26,26,512） -> (208,208,64)

    # f4.shape=(26,26,512) -> 26,26,512
    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍，(26,26,512) -> (52,52,256)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍 (52,52,512) -> (104,104,128)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍，(104,104,128) -> (208,208,64)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 再进行一次卷积，将通道数变为2（要分类的数目） (208,208,64) -> (208,208,2)
    x = Conv2D(class_number, (3, 3), padding='same')(x)
    # reshape: (208,208,2) -> (208*208,2)
    x = Reshape((int(input_height / 2) * int(input_width / 2), -1))(x)
    # 求取概率
    output = Softmax()(x)

    return output


def main(Height=416, Width=416):
    """ model 的主程序，语义分割，分为两部分，第一部分特征提取，第二部分放大图片"""

    # 第一部分 编码，提取特征，图像size减小，通道增加
    img_input, feature_map_list = encoder(input_height=Height, input_width=Height)

    # 第二部分 解码，将图像上采样，size放大，通道减小
    output = decoder(feature_map_list, class_number=class_number, input_height=Height, input_width=Width,
                     encoder_level=3)

    # 构建模型
    model = Model(img_input, output)

    # model.summary()


    return model


if __name__ == '__main__':
    main(Height=416, Width=416)
