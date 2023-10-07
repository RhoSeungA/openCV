import cv2 as cv

# 흑백으로
gray = cv.imread('soccer.jpg',cv.IMREAD_GRAYSCALE)
cv.imshow('original',gray)

gray_mask= cv.inRange(gray,120,170) #120~170 이면 white 아니면 black
cv.imshow('inRange',gray_mask)



# 칼라로
color_img = cv.imread('soccer.jpg')
cv.imshow('original_color',color_img)
# 칼라 공간을 bgr -> hsv로
hsv_img = cv.cvtColor(color_img,cv.COLOR_BGR2HSV)
red_mask= cv.inRange(hsv_img,(-10,50,50),(10,255,255))# 뒤에 두개 기준치 -> 통상적으로 레드는 -10~
cv.imshow('red_mask',red_mask) # red에 해당되는 부분만 하얀색으로 나옴
img_red= cv.bitwise_and(color_img,color_img,mask=red_mask)
cv.imshow('img_red',img_red) #-> 빨강으로 인식되는 부분만 그 원래 색으로 나옴.
# i
cv.waitKey()
cv.destroyAllWindows()