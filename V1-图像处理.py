import cv2
import  matplotlib.pyplot as plt
import numpy as np
#1.读取图片
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# print(img)#输出图片的像素向量组，按照三色原理，每个向量有三个分量；

#imshow显示图像，可以创建多个窗口
# cv2.imshow('girl',img)

#waitkey为等待，0代表按键盘任意键退出，1000代表1000毫秒后退出；
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#读取图片维度
# print(img.shape)

#读取灰度图片
# img = cv2.imread('/Users/violet/Downloads/girl.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('girl',img)
# cv2.waitKey(0)

#查看像素点个数
# print(img.size)


# #2.视频读取
# #Opencv读取视频的大致思路是先判断是否能够打开，如果能打开就循环读取里面的每一帧图片
# vc = cv2.VideoCapture('/Users/violet/Downloads/LOL.mp4')
#
# #判断是否能够读取视频
# if vc.isOpened():
#     Open,frame = vc.read()
# else:
#     Open = False
# while Open:
#     ret,frame = vc.read()
#     if frame is None:
#         break
#     elif ret == True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result',gray)
#         if cv2.waitKey(100) & 0xFF == 27:#暂停100毫秒，接下来100毫秒按下ASCII值为27的按键（ESC）
#             break
# vc.release()
# cv2.destroyAllWindows()

#3.读取部分图像
#读取图片中的某一部分感兴趣窗口，attention机制便是由此发展出来的
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# img2 = img[50:200,100:400]
# cv2.imshow('girl',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#4.通道分离和合并
# #由于图像要表示颜色，所有颜色都可以有三原色进行表示，考虑最基本的三原色，我们可以通过以下方式将他们分开；
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# b,g,r = cv2.split(img)
#
# #也可以将它们合并
# img = cv2.merge((b,g,r))
#
# #可以只看一个通道的图片，蓝、绿、红都试一次
# img1 = img.copy()
# img1[:,:,0] = 0
# img1[:,:,1] = 0
# # img1[:,:,2] = 0
# cv2.imshow('only_green',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#5.填充边界
#卷积网络中经常会用到填充padding，所谓边界填充就是在图像边界处填充一圈像素。
# top_size,bottom_size,left_size,right_size = 50,50,50,50
#
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# replicate = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT,value = 0)
# plt.subplot(231),plt.imshow(img,'gray'),plt.title('orignal')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('replicate')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('reflect')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('reflect101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('wrap')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('constant')
# plt.show()

#6.对像素点的直接加减
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# img_girl = img + 100
# cv2.imshow('girl',img_girl)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #改变图像大小，进行融合
# img = cv2.imread('/Users/violet/Downloads/girl.jpg')
# img_girl = cv2.resize(img,(300,300))
# # cv2.imshow('girl',img_girl)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# img_paper = cv2.imread('/Users/violet/Downloads/IMG20200220190457.jpg')
# img_paper = cv2.resize(img_girl,(300,300))
# res = cv2.addWeighted(img_girl,0.1,img_paper,0.9,0)
# cv2.imshow('resize',res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
