# 히스토그램 평활화하기
import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('mistyroad.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)		# 명암 영상으로 변환하고 출력
plt.imshow(gray,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

h=cv.calcHist([gray],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
# images: Sequence[UMat],channels: Sequence[int],
# mask: UMat | None,histSize: Sequence[int],ranges: Sequence[float],
plt.plot(h,color='r',linewidth=1), plt.show()

#equalizeHist
equal=cv.equalizeHist(gray)			# 히스토그램을 평활화하고 출력
plt.imshow(equal,cmap='gray'), plt.xticks([]), plt.yticks([]), plt.show()

# 다시 히스토그램 계산
h=cv.calcHist([equal],[0],None,[256],[0,256])	# 히스토그램을 구해 출력
plt.plot(h,color='r',linewidth=1), plt.show()