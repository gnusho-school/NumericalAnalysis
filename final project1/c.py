import numpy as np
import os
import glob
import shutil
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.sparse.linalg import svds
from scipy.linalg import svd

n=0
A_tmp=np.zeros((1,1024),float)

def get_image_input():
	global mean_matrix, n, A_tmp
	for root, dirs, files in os.walk('Face'):
		for fname in files:
			full_fname=os.path.join(root, fname)
			if full_fname=="Face\desktop.ini": continue
			tmp=Image.open(full_fname)
			img=tmp.resize((32,32))
			pix=np.array(img)
			pix_=np.reshape(pix,(1,1024))
			A_tmp=np.append(A_tmp,pix_, axis=0)
			n=n+1

def render():
	global n, mean_matrix, A_tmp
	get_image_input()
	
	mean,V=cv2.PCACompute(A_tmp, mean=None, maxComponents=40)
	print(mean.shape)
	print(V.shape)

	for i in range(n):
		A_tmp[i]=A_tmp[i]-mean

	input_image='Faces Gray\Adrian_McPherson_0001.pgm'
	input_tmp=Image.open(input_image)
	input_img=input_tmp.resize((32,32))
	input_pix=np.array(input_img)
	input_pix_=np.reshape(input_pix,(1024))

	input_c=np.zeros(40)
	input_multi=input_pix_-mean

	for i in range (40):
		input_c[i]=np.dot(input_multi,V[i].T)

	mini=A_tmp[0]
	m=10000000
	num=0
	for j in range (n):
		file_c=np.zeros(40)
		for k in range (40):
			file_c[k]=np.dot(A_tmp[j],V[k].T)

		cal=input_c-file_c
		for i in range(40):
			cal[i]=cal[i]/(i+1)
		tmp=np.dot(cal,cal.T)
		if tmp<m:
			m=tmp
			num=j
			mini=A_tmp[j]


	sumation=mean
	for i in range(40):
		sumation=sumation+input_c[i]*V[i]

	sumation=sumation.reshape(32,32)
	plt.imshow(sumation,cmap='gray')
	plt.show()

	cnt=0
	for root, dirs, files in os.walk('Face'):
		for fname in files:
			full_fname=os.path.join(root, fname)
			if full_fname=="Face\desktop.ini": continue
			if cnt==num: 
				print(full_fname)
				break
			cnt=cnt+1

	#mean=mean.reshape(32,32)

if __name__=="__main__":
	render()
