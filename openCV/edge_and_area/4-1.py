import cv2 as cv
img=cv.imread('check.png')
# img=cv.imread('soccer.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 블러 처리를 안하면, 작은 점같은 것들도, 중요한 영역인줄 알고 찾게됨
# 그래서 보통은 에지 찾기 전에 스무딩 처리를 하는 것이 일반적
gray=cv.blur(gray,(3,3)) # 잡은 제거 효과

# sobel 함수 제공해줌  / 영상은 그레이 이미지만 가능함 -> 그레이 이미지로 먼저 만들어줌
# cv.CV_32F : 한바이트 쓰면 음수값까지 안됨.. 그래서 32bit - 4바이트 float 형태로 바꿈.
# 1,0 으로 되어 있으면 : sobel_x
# 0,1 으로 되어 있으면 : sobel_y
grad_x=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3)	# 소벨 연산자 적용, sobel_x
grad_y=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3) #sobel_y

# 라플라시안
grad=cv.Laplacian(gray,cv.CV_32F)

# 우리는 절대값이 크면 중요한 값으로 인식하기로 함.. 근데 막 처리를 안하면 - 값은 다 처리됨
# 그래서 절대값을 취해야함!!

sobel_x=cv.convertScaleAbs(grad_x)	# 절대값을 취해 양수 영상으로 변환 # 주어진 행렬 값을 다 양수로!
sobel_y=cv.convertScaleAbs(grad_y)

lap=cv.convertScaleAbs(grad) #얘도 양소에 대한 처리 해야함!

# 행열 두개 더함, 0.5 0.5 비율로
edge_strength=cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)	# 에지 강도 계산
#0.5*sobel_x + 0.5*sobel_y

cv.imshow('Original',gray)
cv.imshow('sobelx',sobel_x)
cv.imshow('sobely',sobel_y)
cv.imshow('edge strength',edge_strength)
cv.imshow('lap',lap)
cv.waitKey()
cv.destroyAllWindows()