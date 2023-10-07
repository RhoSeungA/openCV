import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED) 
img2=cv.imread('morph.jpg',cv.IMREAD_UNCHANGED)

#t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
t,bin_img=cv.threshold(img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# 이진 영상을 만든 후!
#plt.imshow(bin_img,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()

b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
#plt.imshow(b,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()

se=np.uint8([[0,0,1,0,0],			# 구조 요소
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0]]) # 구조 요소에 포함 시킬거냐?

b_dilation=cv.dilate(b,se,iterations=1)	# 팽창 // 1번 반복!
#plt.imshow(b_dilation,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()

b_erosion=cv.erode(b,se,iterations=1)	# 침식 입력영상, 구조요소, 몇번 반복할건지.
#plt.imshow(b_erosion,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()

b_closing=cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1)	# 닫기 -
# 팽창 후-> 침식!
#plt.imshow(b_closing,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()

b_opening=cv.dilate(cv.erode(b,se,iterations=1),se,iterations=1)	# 닫기 -
# 침식 후 팽창!
#plt.imshow(b_opening,cmap='gray'), plt.xticks([]), plt.yticks([])
#plt.show()


morph = np.hstack((b,b_dilation,b_erosion,b_closing,b_opening))
cv.imshow('Sharpening',morph)
cv.waitKey()
cv.destroyAllWindows()