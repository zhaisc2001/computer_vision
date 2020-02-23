import cv2
import numpy as np
def showimg(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#1.Canny边缘检测算法：最优检测，最优定位准则，检测点和边缘点一一对应；
#算法实现步骤：
#1.使用高斯滤波器，以平滑图像，滤除噪声；
#2.计算出图像中每个点的梯度以及方向；
#3.使用非极大值抑制，消除边缘检测带来的杂散响应；
#4.使用双阈值确定真实的和潜在的边缘；
#5.抑制孤立弱边缘完成边缘检测；

img = cv2.imread('/Users/violet/Downloads/girl.jpg',cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img,80,160)
v2 = cv2.Canny(img,80,200)
res = np.hstack((v1,v2))
showimg('res',res)