import cv2
import copy
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from IPython.display import display

chart0=[]
chart1=[]
chart2=[]
chart3=[]
chart4=[]
chart5=[]

def fun(name):
	
	img_ = cv2.imread(name)
	img = np.array(img_)
	yuv_img = cv2.cvtColor(img_, cv2.COLOR_RGB2YUV)
	yuv=np.array(yuv_img)

	fig=plt.figure()
	rows=2
	cols=3
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]

	y=yuv[:,:,0]
	u=yuv[:,:,1]
	v=yuv[:,:,2]

	tmp=copy.deepcopy(img)
	tmp[:,:,1]=0
	tmp[:,:,2]=0

	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.imshow(tmp)
	ax1.axis("off")

	tmp=copy.deepcopy(img)
	tmp[:,:,0]=0
	tmp[:,:,2]=0

	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.imshow(tmp)
	ax2.axis("off")

	tmp=copy.deepcopy(img)
	tmp[:,:,0]=0
	tmp[:,:,1]=0

	ax3 = fig.add_subplot(rows, cols, 3)
	ax3.imshow(tmp)
	ax3.axis("off")

	#v=cr->red u=cb=>blue
	tmp=copy.deepcopy(yuv)
	tmp[:,:,0]=v
	tmp[:,:,1]=u
	tmp[:,:,2]=u

	ax4 = fig.add_subplot(rows, cols, 4)
	ax4.imshow(tmp)
	ax4.axis("off")


	ax5 = fig.add_subplot(rows, cols, 5)
	ax5.imshow(y,cmap='gray')
	ax5.axis("off")	

	tmp=copy.deepcopy(yuv)
	tmp[:,:,2]=u
	tmp[:,:,0]=v
	tmp[:,:,1]=v

	ax6 = fig.add_subplot(rows, cols, 6)
	ax6.imshow(tmp)
	ax6.axis("off")

	plt.show()

	r=r.flatten()
	b=b.flatten()
	g=g.flatten()
	y=y.flatten()
	u=u.flatten()
	v=v.flatten()

	matrix=np.cov(r,g)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart0.append(p)

	matrix=np.cov(b,g)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart1.append(p)	

	matrix=np.cov(r,b)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart2.append(p)

	matrix=np.cov(y,u)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart3.append(p)

	matrix=np.cov(u,v)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart4.append(p)

	matrix=np.cov(v,y)
	p=matrix[0,1]/(np.sqrt(matrix[0,0])*np.sqrt(matrix[1,1]))
	chart5.append(p)

path_dir='correlation'
file_list1=os.listdir(path_dir)

for i in range(10):
	str='correlation/'+file_list1[i]
	fun(str)

df=pd.DataFrame({"rg":chart0, "gb":chart1, "br":chart2, "yu":chart3, "uv":chart4, "vy":chart5})
display(df)
#tmp=copy.deepcopy(img)
	#tmp[:,:,0]=0
	#tmp[:,:,2]=0

	#plt.imshow(tmp)
	#plt.show()	

