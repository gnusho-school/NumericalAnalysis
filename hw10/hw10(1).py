import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

mean_mat=np.zeros((20,1024))

def fourier(name,a,b,i):
	global mean_mat
	img_ = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img__=cv2.resize(img_, dsize=(128,128))
	img=img__[a:a+64,b:b+64]
	f=np.fft.fft2(img)
	fshift=np.fft.fftshift(f)
	magnitude=32*np.log(np.abs(fshift)+1)
	sum=0
	for k in range(64):
		for j in range(64): sum+=magnitude[k][j]**2
	sum=np.sqrt(sum)
	magnitude=magnitude/sum

	matrix=magnitude[32:64,32:64]
	vector=matrix.flatten()
	tmp=np.dot(vector,vector.T)
	vector=vector*10/np.sqrt(tmp) #값들을 normalization 시킴
	mean_mat[i]+=vector

def decision(name,a,b):
	global mean_mat
	#print(str)
	img_ = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
	img__=cv2.resize(img_, dsize=(128,128))
	img=img__[a:a+64,b:b+64]
	f=np.fft.fft2(img)
	fshift=np.fft.fftshift(f)
	magnitude=32*np.log(np.abs(fshift)+1)
	sum=0
	for k in range(64):
		for j in range(64): sum+=magnitude[k][j]**2
	sum=np.sqrt(sum)
	magnitude=magnitude/sum

	matrix=magnitude[32:64,32:64]
	vector=matrix.flatten()
	tmp=np.dot(vector,vector.T)
	vector=vector*10/np.sqrt(tmp) #값들을 normalization 시킴
	
	mini=1.0
	num=25
	min_n=0
	for i in range(20):
		v=vector-mean_mat[i]
		dist=np.sqrt(np.dot(v,v.T))
		if mini>dist: 
			num=i
			min_n=mini
			mini=dist

	return num,mini,min_n
	

path_dir='data'
file_list1=os.listdir(path_dir)

for i in range(20):
	str='data/'+file_list1[i]
	fourier(str,32,32,i)
	fourier(str,48,48,i)
	fourier(str,16,32,i)
	fourier(str,48,16,i)
	fourier(str,16,16,i)
	mean_mat[i]/=5

path_dir='input'
file_list2=os.listdir(path_dir)

for i in range(23):
	str='input/'+file_list2[i]
	recog, one, two=decision(str,64,64)
	if recog<21:print(file_list2[i],file_list1[recog])
	else:
		print(file_list2[i], "no") 
	#print(one,two)
	#print("\n")
