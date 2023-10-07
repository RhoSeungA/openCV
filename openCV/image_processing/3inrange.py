import cv2 as cv

# gray = cv.imread('soccer.jpg', cv.IMREAD_GRAYSCALE)
# cv.imshow('original - gray', gray)
#
# gray_mask = cv.inRange(gray, 120,170)    # 120~170이면 white, 아니면 black
# cv.imshow('inRange', gray_mask)


img = cv.imread('soccer.jpg')
cv.imshow('original',img)

hsv_img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
red_mask = cv.inRange(hsv_img,(-10,50,50),(10,255,255)) # hsv에서 red의 범위
cv.imshow('red_mask',red_mask) #ㅃㄹ간 영역만 흰색으로 표현됨.

img_red= cv.bitwise_and(img,img,mask=red_mask)
cv.imshow('img_red',img_red)
# A and B : if B=0 -=> 0 else if B=1,A=0



cv.waitKey()
cv.destroyAllWindows()