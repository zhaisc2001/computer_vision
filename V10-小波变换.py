import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pywt

img = cv.imread('/Users/violet/Downloads/girl.jpg')
img = cv.cvtColor(img,cv.COLOR_RGB2GRAY).astype(np.float32)

dwt_result = pywt.dwt2(img,'haar')
L, (H, V, D) = dwt_result

plt.rcParams['font.sans-serif']=['SimHei']
plt.subplot(221),plt.imshow(L,'gray'),plt.title('低频')
plt.subplot(222),plt.imshow(H+255,'gray'),plt.title('水平高频')
plt.subplot(223),plt.imshow(V+255,'gray'),plt.title('垂直高频')
plt.subplot(224),plt.imshow(D+255,'gray'),plt.title('对角高频')
plt.show()
