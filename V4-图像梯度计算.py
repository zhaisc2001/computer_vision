import cv2
import numpy as np

def showimg(image,name):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#1.图像梯度-Sobel算子
# dst = cv2.sobel(img,ddepth,dx,dy,ksize)
#ddepth一般取-1，指图像的深度；
#dx沿水平方向求梯度设置为0或1；
#ksize指Sobel算子大小

img = cv2.imread('/Users/violet/Downloads/girl.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('girl',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#沿水平方向进行梯度提取
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)#cv2.CV_64F表示每一个像素占64位浮点数
# showimg(sobelx,'soblex')
# sobelx = cv2.convertScaleAbs(sobelx)#取绝对值
# showimg(sobelx,'sobelx')

#沿竖直方向进行梯度提取
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
# sobelx = cv2.convertScaleAbs(sobelx)
# showimg(sobelx,'sobelx')

#分别计算x,y再进行求和
# sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
# showimg(sobelxy,'sobelxy')

#不建议直接计算
# sobelxy =cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
# showimg(sobelxy,'sobelxy')

#scchar算子和Laplacian算子
#不同算子的差异
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy =  cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy,scharrxy,laplacian))
showimg(res,'res')