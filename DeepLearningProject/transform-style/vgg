# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import pdb

#一般给的vgg先进行均值初始化通道的是三个颜色数据，所以拿来mean_pixel.下面数据是RGB均值
#是官方的数据
MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

#VGG的net操作,第一个是vgg模型路径。
def net(data_path, input_image):
    #写vgg的层。
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    #加载vgg模型,对于.mat文件用io.loadmat读取。
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))  #内部存储好的结构，文档介绍的。
    weights = data['layers'][0]    #获取所有权重。

    net = {}
    current = input_image
    #坐标和内容,开始进行卷积操作,所以i表示层
    for i, name in enumerate(layers):
        kind = name[:4] #我们关注前四个字母，就能判别出来是哪个层。
        if kind == 'conv':
            #获取w和b,第i个层，其他内容都是0即可，vgg的mat文件就是这样读取。 w,h,个数(in_channel)，通道数(out_channel)
            # 跟tensorflow不一样，
            #所以下面我们需要将w部分进行transpose改变顺序。
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]\
            #根据上文，改变顺序。
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1) #只有一个维度，别忘了reshape
            #组合w和b,在卷积层进行计算,里面是输入，w,b
            current = _conv_layer(current, kernels, bias)
            #下面是各种其他操作
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        #赋值给我们需要生成的网络。
        net[name] = current
    #操作完别忘了判定我们生成的网络层数必须跟我们设定的一样。
    assert len(net) == len(layers)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

#下面是预处理，也就是图像先减去均值的操作，这是图像处理时的必要步骤，类似数据增强。
def preprocess(image):
    return image - MEAN_PIXEL


def unprocess(image):
    return image + MEAN_PIXEL
