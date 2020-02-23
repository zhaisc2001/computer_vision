import cv2
import numpy as np

#1.图像金字塔是多尺度表达的一种，是一种以多分辨率来解释图像的简单有效的结构；
#图像层级越高，图片越小，分辨率越低；
#高斯金字塔：向下采样（缩小），将所有的偶数行和列去掉；
#拉普拉斯金字塔：向上采样，拿0填充，再进行平均；
def showimg(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#上采样：
img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# up = cv2.pyrUp(img)
# showimg('Up',up)
#下采样
# down = cv2.pyrDown(img)
# showimg('down',down)

#2.轮廓检测：边缘与轮廓是不一样的，边缘是离散状态，轮廓是个连续的整体；
#轮廓检测函数cv2.findContours(img,mode,method)
# 参数：
# mode:轮廓检索模式
# RETR_EXTERNAL ：只检索最外面的轮廓；
# RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中；
# RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界;
# RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次;
# method:轮廓逼近方法 。# 通常情况下使用这个。
# CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
# CHAIN_APPROX_SIMPLE:压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。

#为了更高的准确率，我们使用二值图像
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# showimg('gray',thresh)
#接下来用检测函数检测轮廓
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
img_1 = gray.copy()
cnt = contours[0]
res = cv2.drawContours(img_1,[cnt],-1,(0,0,255),2)
showimg('contours',res)