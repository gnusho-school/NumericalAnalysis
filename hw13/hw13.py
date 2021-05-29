#hw13
#y=2*x-1
#12C6=924

import numpy as np
import matplotlib.pyplot as plt
import random

a=2
b=-1

lin=[-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
x=np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
y=a*x+b

noise=np.random.randn(12)*np.sqrt(2)
#print(noise)

yy=y+noise


plt.figure(figsize=(10, 7))
plt.plot(x, y, color='r', label='y = 2*x -1')
plt.scatter(x, yy, label='data')
plt.legend(fontsize=18)
plt.show()

y=yy

x_bar=x.mean()
y_bar=y.mean()

w_12=((x-x_bar)*(y-y_bar)).sum()/((x-x_bar)**2).sum()
b_12=y_bar-w_12*x_bar

print('w_12: {:.2f}'.format(w_12))
print('b_12: {:.2f}'.format(b_12))

criterion=0.2
w_6=10
b_6=10
before_error=10
cnt=0

while True:
	tmp=random.sample(lin,6)
	tmp_x=[]
	tmp_y=[]
	for i in range(6): 
		tmp_x.append(x[tmp[i]+5])
		tmp_y.append(y[tmp[i]+5])
	sample_x=np.array(tmp_x)
	sample_y=np.array(tmp_y)

	x_bar=sample_x.mean()
	y_bar=sample_y.mean()

	w=((sample_x-x_bar)*(sample_y-y_bar)).sum()/((sample_x-x_bar)**2).sum()
	b=y_bar-w*x_bar

	error=(np.abs(w*sample_x+b-sample_y)/np.sqrt(w**2+1)).sum()/6
	#print(error)
	if before_error>error:
		w_6=w
		b_6=b
		before_error=error

	cnt+=1

	if error<criterion or cnt>10000:
		break

print('w_6: {:.2f}'.format(w_6))
print('b_6: {:.2f}'.format(b_6))
print(error)