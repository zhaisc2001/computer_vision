import cv2
import matplotlib.pyplot as plt

#1.阈值分割的基本操作以及滤波函数
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # ret,dst = cv2.threshold(src,thresh,maxval,type)
# #src：输入图，只能输入单通道图片，通常为灰度图片；
# #dst：输出图；
# #thresh:阈值
# #maxval:经过type的方法决定后，对输入图超出（或小于）阈值部分的处理
# ret1,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)#超过阈值部分取maxval，否则取0
# ret2,thresh2 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)#THRESH_BINARY的反转
# ret3,thresh3 = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)#大于阈值设为阈值，否则不变；
# ret4,thresh4 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)#大于阈值部分不改变，否则设为0；
# ret5,thresh5 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)#上述的反面；
# titles = ['ORIGNAL IMAGE','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#2.滤波函数
img = cv2.imread('/Users/violet/Downloads/girl.jpg')

# # 均值滤波，输入图像与卷积核大小
# blur = cv2.blur(img,(3,3))
# cv2.imshow('均值滤波',blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #方框滤波：基本和均值滤波一样，可以选择归一化使normalize = True,这样和均值滤波一样，
# #如果normalize = False,那么大部分都将发生越界的情况；
# box = cv2.boxFilter(img,-1,(3,3),normalize=True)
# cv2.imshow('box',box)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #高斯滤波：高斯滤波的卷积核是满足高斯分布的
# aussian = cv2.GaussianBlur(img,(5,5),1)
# cv2.imshow('Gaussian',aussian)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #中值滤波：相当于用中值替代所有
# median = cv2.medianBlur(img,5)
# cv2.imshow('median',median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()