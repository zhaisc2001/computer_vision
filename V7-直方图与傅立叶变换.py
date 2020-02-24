import cv2
import numpy as np
import matplotlib.pyplot as plt

def showimg(name,image):
    cv2.imshow()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#1.calcHist()函数解释
img = cv2.imread('/Users/violet/Downloads/girl.jpg',0)
# cv2.calcHist([img],[0],None,[256],[0,256])
#calcHist()函数解释
#cv2.calcHist(images,channels,mask,histSize,ranges)
#images:原图像图像格式为uint8或float32。当传入函数时应用中括号[]
#channels:同样用中括号，如果传入图像是灰度图，它的值就是[0]，如果是
#彩色图像的传入的参数可以是[0][1][2]，它们分别对应着BGR。相当于统计不同的
#通道的直方图。
#mask:掩模图像。统计整幅图像直方图就把它当作None。但是如果你想统计图像局部的
#直方图你就的只做一个掩模图像，并将它作用在这个图像上，就会统计掩模图像；
#histSize：BIN的数目，表示你需要统计的像素范围。选择[0-10]范围的像素统计频率。
#ranges:像素值范围常为[0,256]
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.hist(img.ravel(),256)
# plt.show()

# mask = np.zeros(img.shape[:2],np.uint8)
# mask[100:300,100:400] = 255
# maked_img = cv2.bitwise_and(img,img,mask=mask)
#
# hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
# hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])
#
# plt.subplot(221),plt.imshow(img,'gray')
# plt.subplot(222),plt.imshow(mask,'gray')
# plt.subplot(223),plt.imshow(maked_img,'gray')
# plt.subplot(224),plt.imshow(hist_full),plt.plot(hist_mask)
# plt.xlim([0,256])
# plt.show()

#2.傅立叶变换
# img_float32 = np.float32(img)
# dft =cv2.dft(img_float32,flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)#np中shift变换，将其转到中心
# #得到灰度图能表示的形式
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
# plt.subplot(121),plt.imshow(img,cmap='gray')
# plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum,cmap='gray')
# plt.title('Magnitude Spectrum'),plt.xticks([]),plt.yticks([])
# plt.show()

#傅立叶滤波
img_float32 = np.float32(img)

dft = cv2.dft(img_float32,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows,cols = img.shape
crow,ccol = int(rows/2),int(cols/2)

#高通滤波
mask = np.ones((rows,cols,2),np.uint8)
mask[crow-30:crow+30,ccol-30:ccol+30] = 0

#IDFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('Input Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(img_back,cmap='gray')
plt.title('Result'),plt.xticks([]),plt.yticks([])

plt.show()