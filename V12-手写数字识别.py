# -*- coding:utf-8 -*-
"""
@author:UESTC_Sicca
@file:V12-手写数字识别.py
@time:2020/11/9
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
import time

# 训练集文件
train_images_idx3_ubyte_file = '/Users/violet/Documents/计算机视觉/MNIST_Database/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = '/Users/violet/Documents/计算机视觉/MNIST_Database/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = '/Users/violet/Documents/计算机视觉/MNIST_Database/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = '/Users/violet/Documents/计算机视觉/MNIST_Database/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)  #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    print(offset)
    fmt_image = '>' + str(image_size) + 'B'  #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    #plt.figure()
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        #print(images[i])
        offset += struct.calcsize(fmt_image)
#        plt.imshow(images[i],'gray')
#        plt.pause(0.00001)
#        plt.show()
    #plt.show()

    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为模样数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_data():
    '''
    :return: 依次为训练样本及其标签，测试样本及其标签
    '''
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    train_images = train_images/255
    test_images = test_images/255
    #
    #
    # # 查看前十个数据及其标签以读取是否正确
    # plt.figure(figsize=(2,5))
    # for i in range(10):
    #     plt.subplot(2, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(train_images[i])
    #     plt.xlabel(train_labels[i])
    #     print(train_labels[i])
    # plt.show()
    return train_images,train_labels,test_images,test_labels

def Softmax(X):
    ex = np.exp(X)
    return ex/np.sum(ex)


def one_hot_label(label):
    lab = np.zeros((label.size, 10, 1))
    for i in range(len(lab)):
        lab[i][int(label[i])] = 1
    return lab

train_images,train_labels,test_images,test_labels = load_data()
train_labels = one_hot_label(train_labels)
test_labels = one_hot_label(test_labels)

time_start = time.time()
W1 = np.random.randn(9,9,20)
W3 = (2*np.random.rand(100,2000)-1)/20
W4 = (2*np.random.rand(10,100)-1)/10
alpha=0.01
# S_dW1,S_dW3,S_dW4 = 0,0,0
for i in range(len(train_images)):
    V1 = np.zeros((20,20,20))
    for k in range(20):
        V1[:,:,k] = correlate2d(train_images[i],W1[:,:,k],'valid')
    Y1 = np.maximum(0,V1)
    Y2 = (Y1[::2,::2] + Y1[1::2,::2] + Y1[::2,1::2] + Y1[1::2,1::2])/4

    y2 = np.reshape(Y2,[2000,1])
    V3 = np.dot(W3,y2)
    Y3 = np.maximum(0,V3)
    V = np.dot(W4,Y3)
    # if i == 11:print(Y)
    Y = Softmax(V)
    d = train_labels[i]

    e = d-Y
    delta = e
    e3 = np.dot(W4.T,delta)
    delta3 = (V3>0)*e3
    e2 = np.dot(W3.T,delta3)
    dW4 = np.dot(alpha*delta,Y3.T)
    dW3 = np.dot(alpha*delta3,y2.T)

    E2 = np.reshape(e2,[10,10,20])
    E1 = np.zeros(np.shape(Y1))
    E2_4 = E2/4
    E1[::2,::2,:] = E2_4
    E1[::2,1::2,:] = E2_4
    E1[1::2,::2,:] = E2_4
    E1[1::2,1::2,:] = E2_4
    delta1 = (V1>0)*E1
    dW1 = np.zeros(np.shape(W1))
    for k in range(20):
        dW1[:,:,k] = np.dot(alpha,correlate2d(train_images[i],delta1[:,:,k],'valid'))
    # if (i+1)%100 == 0:
    W1 += dW1
    W3 += dW3
    W4 += dW4
        # S_dW1,S_dW2,S_dW3 = 0,0,0
    # else:
    #     S_dW1 += dW1
    #     S_dW3 += dW3
    #     S_dW4 += dW4

    print("已训练"+str((i+1)/len(train_images)*100)+"%")
time_end = time.time()

s = 0
for i in range(len(test_images)):
    V1 = np.zeros((20, 20, 20))
    for k in range(20):
        V1[:, :, k]=correlate2d(test_images[i], W1[:, :, k], 'valid')
    Y1 = np.maximum(0, V1)
    Y2 = (Y1[::2, ::2] + Y1[1::2, ::2] + Y1[::2, 1::2] + Y1[1::2, 1::2]) / 4

    y2 = np.reshape(Y2, [2000, 1])
    V3 = np.dot(W3, y2)
    Y3 = np.maximum(0, V3)
    V = np.dot(W4, Y3)
    Y = Softmax(V)

    Y_pred = np.argmax(Y)
    Y_test = np.argmax(test_labels[i])
    if Y_pred != Y_test:s+=1
print("训练用时为",time_end-time_start,"秒")
print("正确率为"+str((1-s/len(test_labels))*100)+"%")