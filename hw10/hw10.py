import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def fourier(name):
	img_ = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img__=cv2.resize(img_, dsize=(128,128))
	img=img__[32:96,32:96]
	f=np.fft.fft2(img)
	fshift=np.fft.fftshift(f)
	magnitude=32*np.log(np.abs(fshift))

	rows,cols=img.shape
	crow,ccol=int(rows/2), int(cols/2)

	#fshift[crow-30:crow+30, ccol-30:ccol+30]=0
	f_ishift=np.fft.ifftshift(fshift)
	img_back=np.fft.ifft2(f_ishift)
	img_back=np.abs(img_back) 

	plt.subplot(131), plt.imshow(img, cmap='gray')
	plt.title('original'), plt.xticks([]), plt.yticks([])
	plt.subplot(132), plt.imshow(magnitude, cmap='gray')
	plt.title('magnitude'), plt.xticks([]), plt.yticks([])
	plt.subplot(133), plt.imshow(img_back, cmap='gray')
	plt.title('after'), plt.xticks([]), plt.yticks([])

	plt.show()

path_dir='data'
file_list=os.listdir(path_dir)
print(file_list)
for i in range(20):
	str='data/'+file_list[i]
	fourier(str)


#64x64 dft => 계수들 분포의 절반만 이용하면 됨
#일부 영역만 뽑아서 비교 / low frequency dc 성분 제외한 ac성분 위주로 비교
#dominant한 영역에 따라서 비교
#dft하는 영역에 따라서 magnitude가 달라짐 
#scale을 잘 맞추는게 중요함 => 패턴을 다 포함할 수 있도록
#rotation 시키면 완전히 다른 패턴이 되버리니까 주의 
#vector distance 만 사용해도 잘 될거임
#중요한 계수를 골라서 그정도만 비교해도 됨