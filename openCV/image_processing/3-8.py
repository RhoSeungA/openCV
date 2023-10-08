import cv2 as cv

img=cv.imread('rose.png')
patch=img[250:350,170:270,:]

img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3)
#patch 영상 자름, 색상은 다가져옴
#가로로 5배, 세로로 5배 확대, interpolation => inter_nearest 로 색상 결정하겠다.
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)#cv.INTER_NEAREST
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR) # 계단현상 완화
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

# 이것도 내가 원하는 필터_ 변환행렬을 직접 넣어줄 수 있음 -> affine 변환 행렬을 통해.


cv.imshow('Original',img)
cv.imshow('Resize nearest',patch1) 
cv.imshow('Resize bilinear',patch2) 
cv.imshow('Resize bicubic',patch3) 

cv.waitKey()
cv.destroyAllWindows()