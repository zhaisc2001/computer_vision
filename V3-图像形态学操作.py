import cv2
import numpy as np

#1.形态学-腐蚀操作
#通常都是通过二值图片进行腐蚀操作，腐蚀大概意思就是往里面缩一点；
img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# cv2.imshow('girl',img)
# kernel = np.ones((3,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations=1)#迭代次数表示做几次腐蚀操作；
# cv2.imshow('corrosion',erosion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2.形态学-膨胀操作
#通常也是二值图片，膨胀的大概意思是向外面扩张一些；
# kernel = np.ones((3,3),np.uint8)
# dil = cv2.dilate(img,kernel,iterations=3)
# cv2.imshow('dilate',dil)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#3.开运算
#先腐蚀，再膨胀，这个操作能够将主体被腐蚀的部分补偿回去，但是边缘的毛刺能够很好的被清理掉；
# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# cv2.imshow('opening',opening)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#4.闭运算
#先膨胀，再腐蚀；
# kernel = np.ones((5,5),np.uint8)
# closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
# cv2.imshow('closing',closing)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#5.梯度运算
#梯度 = 膨胀-腐蚀，这是另一种计算图像边缘的方法
# kernel = np.ones((2,2),np.uint8)
# gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
# cv2.imshow('gradient',gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#6.礼帽与黑帽
#礼帽=原始输入-开运算结果，黑帽=闭运算-原始输入
# kernel = np.ones((7,7),np.uint8)
# tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
# cv2.imshow('tophat',tophat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# kernel = np.ones((7,7),np.uint8)
# blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
# cv2.imshow('blackhat',blackhat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()