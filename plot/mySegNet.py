#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2020/6/26 15:54
# @Author : v
# @File : mySegNet.py
# @Software: PyCharm
import sys

sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # 输入
    to_input('./picture/127.jpg', width=2, height=12),
    # to_input('./picture/xzlabel_0_2_json_img.png', width=12, height=12),
    #  """编码部分"""
    # 第一个单元
    # to_Conv("conv1", 256, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),

    to_Conv("conv1", 256, 64, offset="(0,0,0)", to="(0,0,0)", height=64, depth=64, width=2),
    to_Conv("conv2", 256, 64, offset="(0,0,0)", to="(conv1-east)", height=64, depth=64, width=2),
    to_Pool("pool1", offset="(0,0,0)", to="(conv2-east)", width=2, height=32, depth=32),

    # 第二个单元
    to_Conv("conv3", 128, 128, offset="(2,0,0)", to="(pool1-east)", height=32, depth=32, width=4),
    to_Conv("conv4", 128, 128, offset="(0,0,0)", to="(conv3-east)", height=32, depth=32, width=4),
    to_Pool("pool2", offset="(0,0,0)", to="(conv4-east)", width=4, height=16, depth=16),

    # 第三个单元
    to_Conv("conv5", 64, 256, offset="(2,0,0)", to="(pool2-east)", height=16, depth=16, width=8),
    to_Conv("conv6", 64, 256, offset="(0,0,0)", to="(conv5-east)", height=16, depth=16, width=8),
    to_Conv("conv7", 64, 256, offset="(0,0,0)", to="(conv6-east)", height=16, depth=16, width=8),
    to_Pool("pool3", offset="(0,0,0)", to="(conv7-east)", width=8, height=8, depth=8),

    # 第四个单元
    to_Conv("conv8", 32, 512, offset="(2,0,0)", to="(pool3-east)", height=8, depth=8, width=16),
    to_Conv("conv9", 32, 512, offset="(0,0,0)", to="(conv8-east)", height=8, depth=8, width=16),
    to_Conv("conv10", 32, 512, offset="(0,0,0)", to="(conv9-east)", height=8, depth=8, width=16),
    to_Pool("pool4", offset="(0,0,0)", to="(conv10-east)", width=16, height=4, depth=4),

    # """解码部分"""
    to_Conv("conv11", 16, 512, offset="(2,0,0)", to="(pool4-east)", height=4, depth=4, width=16),

    # 链接
    to_connection("pool4", "conv11"),
    # 第一个单元
    to_UnPool("unpool1", offset="(2,0,0)", to="(conv11-east)", height=8, depth=8, width=16),
    to_Conv("conv12", 32, 256, offset="(0,0,0)", to="(unpool1-east)", height=8, depth=8, width=8),

    # 第二个单元
    to_UnPool("unpool2", offset="(2,0,0)", to="(conv12-east)", height=16, depth=16, width=8),
    to_Conv("conv13", 64, 128, offset="(0,0,0)", to="(unpool2-east)", height=16, depth=16, width=4),

    # 第三个单元
    to_UnPool("unpool3", offset="(2,0,0)", to="(conv13-east)", height=32, depth=32, width=4),
    to_Conv("conv14", 128, 64, offset="(0,0,0)", to="(unpool3-east)", height=32, depth=32, width=2),

    # 第四个单元
    to_Conv("conv15", 128, 2, offset="(2,0,0)", to="(conv14-east)", height=32, depth=32, width=1),

    # to_Conv("conv2", 128, 64, offset="(1_raster,0,0)", to="(pool1-east)", height=32, depth=32, width=2),
    # to_connection("pool1", "conv2"),
    # to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1_raster),
    to_SoftMax("soft1", 128, "(2,0,0)", "(conv15-east)", caption="SOFT", width=1, height=32, depth=32),
    # to_connection("pool2", "soft1"),
    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
