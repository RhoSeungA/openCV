import cv2 as cv
import sys
import numpy as np

img=cv.imread('soccer.jpg')

#1
gray=cv.imread('soccer.jpg', cv.IMREAD_GRAYSCALE) # BGR 컬러 영상을 명암 영상으로 변환하여 저장 -> imread + cvtColor

if gray is None:
    sys.exit('파일을 찾을 수 없습니다.')

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)	# BGR 컬러 영상을 명암 영상으로 변환 -> 7행 보면, 그냥 바로 grayscale로 읽음

gray_small=cv.resize(gray,dsize=(0,0),fx=0.25,fy=0.25) # 1/4로 줄임

cv.imshow('soccer-gray',gray_small)
print(gray_small.shape)

#사칙 연산!

#2 함수 사용
img_plus = cv.add(gray_small, 50)         # y = x + 50 -> 밝아짐
img_minus = cv.subtract(gray_small, 50)   # y = x - 50  -> 어두워짐
img_multi = cv.multiply(gray_small, 2)    # y = 2 * x  -> 대비가 커짐
img_div = cv.divide(gray_small, 2)        # y = x / 2  -> 대비 약해짐
pp=np.hstack((img_plus, img_minus, img_multi, img_div)) # 쭉 하나로 붙이기.
cv.imshow('point processing',pp)

#3 만약 그냥 이런식으로 연산을 한다면!! -> 모든 값에 50더하고,, 50 빼고,,

img_plus2 = gray_small+50       # y = x + 50 // 배열에 50을 더함 -> 배열의 모든 값에 50씩 더한다.
# 자연스럽게 처리를 못함. 255 더했을때 클램핑 처리가 안돼서 막 까만 부분도 있고..
img_minus2 = gray_small-50      # y = x - 50
img_multi2 = gray_small*2       # y = 2 * x
img_div2 = gray_small/2         # y = x / 2
cv.imshow('test -plus ',img_plus2)
print(gray_small[100,100],img_plus[100,100],img_plus2[100,100]) # 62, 122, 122
print(gray_small[180,80],img_plus[180,80],img_plus2[180,80]) #226 ,255,20 (20 : 오버플로우.. 엉뚱한 값이 나옴)

#4
# cv.imshow('test',img_plus2)
# print(gray_small[100,100], img_plus[100,100], img_plus2[100,100]) # 72
# print(gray_small[180,80], img_plus[180,80], img_plus2[180,80]) # 226

#5
#cv.imshow('test',img_minus2)
#print(gray_small[100,100], img_minus[100,100], img_minus2[100,100]) # 72
#print(gray_small[120,180], img_minus[120,180], img_minus2[120,180]) # 27

#6 두 영상 합치기
img512=cv.resize(img,dsize=(512,512)) #크기 조절
opencv_img=cv.imread('opencv_logo512.png')
# 숫자 대신 img 를 더함 (img도 결국 숫자)
img_plus3 = cv.add(img512, opencv_img) # 이미지 더함. 0~255 0~255 : 0~510이렇게 됨.. 더 크면 화이트 처리??
cv.imshow('two images - add',img_plus3)

#7 무조건 더하는 것이 아닌, 비율적으로 더함!! -> addWeigthed 함수
img_plus4 = cv.addWeighted(img512, 0.5, opencv_img, 0.5, 0) #두개를 더함 근데, 비율적으로 더함. addWeighted , 마지막은 그냥 0
cv.imshow('two images - addWeighted',img_plus4)

#8 역변환
img_rev = cv.subtract(255,gray_small)     # y = 255 - x -> 반전 변환 함수
cv.imshow('reverse image',img_rev) # 하얀색이 검은색으로..
#
#9 이진화
# threshold 리턴 값은 두개 -> ret: true/false
# 영상, 임계값 50 , max , type
ret, img_binary50 = cv.threshold(gray_small, 50, 255, cv.THRESH_BINARY)
cv.imshow('threshold1',img_binary50)

# 임계치 다르게
ret, img_binary100 = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY) # 임계치 100
ret, img_binary150 = cv.threshold(gray_small, 150, 255, cv.THRESH_BINARY) # 임계치 150
ret, img_binary200 = cv.threshold(gray_small, 200, 255, cv.THRESH_BINARY)
img_binary=np.hstack((img_binary50, img_binary100, img_binary150, img_binary200))
cv.imshow('threshold2',img_binary)

# 다양한 타입
ret, img_binaryB = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY)
ret, img_binaryBINV = cv.threshold(gray_small, 100, 255, cv.THRESH_BINARY_INV)
ret, img_binaryT = cv.threshold(gray_small, 100, 255, cv.THRESH_TRUNC)
ret, img_binaryT0 = cv.threshold(gray_small, 100, 255, cv.THRESH_TOZERO)
ret, img_binaryT0INV = cv.threshold(gray_small, 100, 255, cv.THRESH_TOZERO_INV) #100보다 크면 블랙 작으면 자기 자신.
img_binary2=np.hstack((img_binaryB, img_binaryBINV, img_binaryT, img_binaryT0, img_binaryT0INV))
cv.imshow('threshold3',img_binary2)

cv.waitKey()
cv.destroyAllWindows()