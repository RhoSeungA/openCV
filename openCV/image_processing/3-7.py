import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.putText(gray,'soccer',(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv.imshow('Original',gray)

# 가우시안 필터( ?, 적용할 필터의 크기 , 0.0 : 시그마 값 얼만큼 부드럽게 할것인가)
smooth=np.hstack((cv.GaussianBlur(gray,(5,5),0.0),cv.GaussianBlur(gray,(9,9),0.0),cv.GaussianBlur(gray,(15,15),0.0)))
cv.imshow('Smooth',smooth) # 필터 크기 커질수록 블러링 정도 up.


#엠보싱 필터 -> 이건 제공안함.
femboss=np.array([[-1.0, 0.0, 0.0],
                  [ 0.0, 0.0, 0.0],
                  [ 0.0, 0.0, 1.0]])

gray16=np.int16(gray) # gray는 한바이트 (8비트) -> 16비트로 늘려줌.
# // 지금까지 썼던 그레이는 8비트였음. 근데 계산 하다 보면,, 엠보싱인 경우
# 젤 차이 크게 나는거 -255~255 .. 그래서 일단 데이터 크기 늘려줌

emboss=np.uint8(np.clip(cv.filter2D(gray16,-1,femboss)+128,0,255))
# 이건 클램핑 처리를 해줌 -> clip 함수를 이용 0보다 작으면 0으로, 255보다 크면 255로

emboss_bad=np.uint8(cv.filter2D(gray16,-1,femboss)+128)
# -255~255되어져 있는 것들, but 우리는 0~255 밖에 표현 안됨
# 그래서 표현해줄수 있는 것의 중앙에 오게끔 +128을 해줌
# -127~383 에 해당되는 범위로 바꿔줌 .
# 근데, 자세히 보면, 부분부분 값들을 처리 못한게 있음. 왜냐면 아직 범위 밖으로 나가기 때문
emboss_worse=cv.filter2D(gray,-1,femboss) # 완전 깜깜한 올록볼록으로 나옴
# -> 그냥 경계에 대한 부분임.. -255가 그냥 다 블랙 처리 되다 보니 올바르지 않게 나옴.



#cv.imshow('Emboss',emboss)
cv.imshow('Emboss_bad',emboss_bad)
cv.imshow('Emboss_worse',emboss_worse)

faverage=np.array([[1.0/9.0, 1.0/9.0, 1.0/9.0],
                  [ 1.0/9.0, 1.0/9.0, 1.0/9.0],
                  [ 1.0/9.0, 1.0/9.0, 1.0/9.0]])
fsharpening1=np.array([[0.0, -1.0, 0.0],
                  [ -1.0, 4.0, -1.0],
                  [ 0.0, -1.0, 0.0]])
fsharpening2=np.array([[0.0, -1.0, 0.0],
                  [ -1.0, 5.0, -1.0],
                  [ 0.0, -1.0, 0.0]])
sharpening3=np.array([[-1.0, -1.0, -1.0],
                  [ -1.0, 8.0, -1.0],
                  [ -1.0, -1.0, -1.0]])
fsharpening4=np.array([[-1.0, -1.0, -1.0],
                  [ -1.0, 9.0, -1.0],
                  [ -1.0, -1.0, -1.0]])

result = cv.filter2D(gray, -1, fsharpening1) # 마지막이 필터 값
#cv.imshow('result', result)

# opencv 에서 제공하는...blur 함수 , medianBlur 함수 , bilateralFilter

gray=cv.imread('coins.png', cv.IMREAD_GRAYSCALE)
average = cv.blur(gray,(9,9))  #두번째 인자는 필터의 크기! 필터 크기 커지면 블러링 up    # 평균값 필터

# average = cv.filter2D(gray, -1, faverage) # cv.blur와 동일한 결과
cv.imshow('result -average', average)


median = cv.medianBlur(gray,9) # 3넣으면 3x3으로      # 중간값 필터 ( 두번째 인자는 필터 크기 3-> 3,ㅌ)
cv.imshow('result - median', median) # 이건 약간 경계가 너무 흐려지지 않음

# 에지의 값은 스무딩 시키지 않겠다.
bilateral = cv.bilateralFilter(gray, -1, sigmaColor=5, sigmaSpace=5)
# 두번째 인자 -> 필터의 크기르 ㄹ어떻게 할거냐. #sigmaColor
# 각 픽셀과 주변요소들로부터 가중 평균을 구함 => 가우시안과 유사
# 단, 픽셀값의 차이도 같이 사용하여 유사한 픽셀에 더 큰 가중치를 두는 방법
# 경계선을 유지하며 스무딩

# 값의 차이가 너무 나면, 평균 구할때 포함시키지 않겠다? -> 경계 -> 경계는 그냥 유지

cv.imshow('result - bilateral', bilateral)

cv.waitKey()
cv.destroyAllWindows()