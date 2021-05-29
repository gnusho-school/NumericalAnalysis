#term project
#pattern recognition
#simulation approach
#vector data들을 random하게 generated (Gaussian distribution)
#K means Clustering
#Test for randomly generated vectors 올바르게 clustering 되는지 하나하나 알고리즘 적으로 비교
#pattern recognitino은 할 때 결국엔 vector를 쓸 것이다

#gaussian 5class의 sample data를 random하게 generate
#X,Y,Z의 gaussian을 다 다르게 만듬
#class마다의 x,y,z의 m값을 다 다르게 만들고 분산도 다양하게 하기
#300samples per each class
#K means clustering을 실행함
#적당히 overlap이 되게하고 적절히 clustering하기

#cluster vector를 확인하고 어디랑 가장 가까운지 확인하기
#maximum distance를 정해주기-> sample일부가 버려지긴 할 것임
#정답이 없으니 rule을 가지고 하기-> 최대한 잘 인지될 수 있도록

#각 distribution마다 100개씩 만들어서 잘 assign되는지 확인하기
#거기에다가 다른 distribution을 가지는 100개의 data가 정의되지 않는다!

#analysis-> 어떤 distribution의 accuracy가 되고 안되고를 분석 아마 cluster들이 연관되있을 것
#m과 분산에 따라서 갈릴 것임
import random
import numpy as np    
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy.linalg as lin

class_=np.zeros((1500,3))

mean_x=[random.randrange(0,10) for i in range(6)]
mean_y=[random.randrange(0,10) for i in range(6)]
mean_z=[random.randrange(0,10) for i in range(6)]

sigma_x=[random.randrange(0,10) for i in range(6)]
sigma_y=[random.randrange(0,10) for i in range(6)]
sigma_z=[random.randrange(0,10) for i in range(6)]

print("mean")
print("x: ",mean_x)
print("y: ",mean_y)
print("z: ",mean_z)
print("")
print("sigma")
print("x: ",sigma_x)
print("y: ",sigma_y)
print("z: ",sigma_z)

# data for training
x1=np.random.randn(300)*np.sqrt(sigma_x[0])+mean_x[0]
y1=np.random.randn(300)*np.sqrt(sigma_y[0])+mean_y[0]
z1=np.random.randn(300)*np.sqrt(sigma_z[0])+mean_z[0]

x2=np.random.randn(300)*np.sqrt(sigma_x[1])+mean_x[1]
y2=np.random.randn(300)*np.sqrt(sigma_y[1])+mean_y[1]
z2=np.random.randn(300)*np.sqrt(sigma_z[1])+mean_z[1]

x3=np.random.randn(300)*np.sqrt(sigma_x[2])+mean_x[2]
y3=np.random.randn(300)*np.sqrt(sigma_y[2])+mean_y[2]
z3=np.random.randn(300)*np.sqrt(sigma_z[2])+mean_z[2]

x4=np.random.randn(300)*np.sqrt(sigma_x[3])+mean_x[3]
y4=np.random.randn(300)*np.sqrt(sigma_y[3])+mean_y[3]
z4=np.random.randn(300)*np.sqrt(sigma_z[3])+mean_z[3]

x5=np.random.randn(300)*np.sqrt(sigma_x[4])+mean_x[4]
y5=np.random.randn(300)*np.sqrt(sigma_y[4])+mean_y[4]
z5=np.random.randn(300)*np.sqrt(sigma_z[4])+mean_z[4]

x6=np.random.randn(300)*np.sqrt(sigma_x[5])+mean_x[5]
y6=np.random.randn(300)*np.sqrt(sigma_y[5])+mean_y[5]
z6=np.random.randn(300)*np.sqrt(sigma_z[5])+mean_z[5]

class_[:300,0]=x1
class_[:300,1]=y1
class_[:300,2]=z1
class_[300:600,0]=x2
class_[300:600,1]=y2
class_[300:600,2]=z2
class_[600:900,0]=x3
class_[600:900,1]=y3
class_[600:900,2]=z3
class_[900:1200,0]=x4
class_[900:1200,1]=y4
class_[900:1200,2]=z4
class_[1200:1500,0]=x5
class_[1200:1500,1]=y5
class_[1200:1500,2]=z5

km = KMeans(n_clusters=5)
km.fit(class_)

label=km.labels_
cc = km.cluster_centers_    
#print(cc)        

test_=np.zeros((600,3))

#data for testing
xx1=np.random.randn(100)*np.sqrt(sigma_x[0])+mean_x[0]
yy1=np.random.randn(100)*np.sqrt(sigma_y[0])+mean_y[0]
zz1=np.random.randn(100)*np.sqrt(sigma_z[0])+mean_z[0]

xx2=np.random.randn(100)*np.sqrt(sigma_x[1])+mean_x[1]
yy2=np.random.randn(100)*np.sqrt(sigma_y[1])+mean_y[1]
zz2=np.random.randn(100)*np.sqrt(sigma_z[1])+mean_z[1]

xx3=np.random.randn(100)*np.sqrt(sigma_x[2])+mean_x[2]
yy3=np.random.randn(100)*np.sqrt(sigma_y[2])+mean_y[2]
zz3=np.random.randn(100)*np.sqrt(sigma_z[2])+mean_z[2]

xx4=np.random.randn(100)*np.sqrt(sigma_x[3])+mean_x[3]
yy4=np.random.randn(100)*np.sqrt(sigma_y[3])+mean_y[3]
zz4=np.random.randn(100)*np.sqrt(sigma_z[3])+mean_z[3]

xx5=np.random.randn(100)*np.sqrt(sigma_x[4])+mean_x[4]
yy5=np.random.randn(100)*np.sqrt(sigma_y[4])+mean_y[4]
zz5=np.random.randn(100)*np.sqrt(sigma_z[4])+mean_z[4]

xx6=np.random.randn(100)*np.sqrt(sigma_x[5])+mean_x[5]
yy6=np.random.randn(100)*np.sqrt(sigma_y[5])+mean_y[5]
zz6=np.random.randn(100)*np.sqrt(sigma_z[5])+mean_z[5]

#test_
test_[:100,0]=xx1
test_[:100,1]=yy1
test_[:100,2]=zz1
test_[100:200,0]=xx2
test_[100:200,1]=yy2
test_[100:200,2]=zz2
test_[200:300,0]=xx3
test_[200:300,1]=yy3
test_[200:300,2]=zz3
test_[300:400,0]=xx4
test_[300:400,1]=yy4
test_[300:400,2]=zz4
test_[400:500,0]=xx5
test_[400:500,1]=yy5
test_[400:500,2]=zz5
test_[500:600,0]=xx6
test_[500:600,1]=yy6
test_[500:600,2]=zz6

library_predict=km.predict(test_)
library_predict[500:600]=-1
#mean들 사이의 거리 계산
dist=np.zeros((5,5))
for i in range(5):
	for j in range(5):
		d=np.sqrt(np.dot(cc[i]-cc[j],cc[i]-cc[j]))
		dist[i][j]=d/2

#print("\n cluster centers")
#print(cc,"\n")
print("dist between")
print(dist )
#거리상으로 하려는데 이럴때는 겹치는 부분이 많을수록 문제가 많이 생김
#그러면 어느정도 많이 겹치면 다른 방법을 쓰자!
#covariace에서 값들을 루트 씌운건 그 방향으로의 축의 길이를 의미함
#그렇다면 이를 통해서 타원의 관계식과 이 거리보다 멀다면 out이라고 할 수도 있을 것!

#cov=np.cov(class_.T)
#e_val,e_vec=lin.eig(cov)
#print(e_val)
#print(e_vec)

e_val=np.zeros((5,3))
e_vec=np.zeros((5,3,3))
cmp_dist=np.zeros(5)
t=0
for i in range(5):
	tmp=class_[t:t+300,:]
	t=t+300
	cov=np.cov(tmp.T)
	tmp_eval,tmp_evec=lin.eig(cov)
	e_val[i]=tmp_eval
	e_vec[i]=tmp_evec
	cmp_dist[i]=max(tmp_eval)

print("")
print("cmp_dist")
print(cmp_dist)
label_tmp=np.zeros((5,6))
labels=np.zeros(6)
t=0
for i in range(5):
	for j in range(0,300):
		k=label[t+j]
		label_tmp[i][k]+=1

	use=-1
	use_cnt=-1
	for j in range(5):
		if use_cnt<label_tmp[i][j]: 
			use=j
			use_cnt=label_tmp[i][j]

	labels[i]=use
	t=t+300
labels[5]=-1
#print(labels)

my_predict=np.zeros(600)
#print(labels[5])
t=0
cnt=0
cnt_all=0
cnt_predict=0
for k in range(6):
	pre_tmp=np.zeros(6)
	cnt=0
	for i in range(k*100,k*100+100):
		test=test_[i]
		dist=1e9
		pick=-1
		for j in range(5):
			tmp=np.sqrt(np.dot(test-cc[j],test-cc[j]))
			if tmp<dist:
				dist=tmp
				pick=j
		if dist<cmp_dist[pick]: my_predict[i]=pick
		else: my_predict[i]=-1
		pre_tmp[int(my_predict[i])]+=1
	#print(i,labels[int(k)])
		if my_predict[i]==labels[k]: cnt=cnt+1
		if my_predict[i]==library_predict[i]: cnt_predict+=1
	print("\n for all",pre_tmp)
	print(k,", ",labels[k],": ",cnt)
	cnt_all+=cnt
print("total: ",cnt_all/6)
print("library: ", cnt_predict/6)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1,y1,z1, marker='o', s=15, cmap='Greens')
ax.scatter(x2,y2,z2, marker='o', s=15, cmap='Greens')
ax.scatter(x3,y3,z3, marker='o', s=15, cmap='Greens')
ax.scatter(x4,y4,z4, marker='o', s=15, cmap='Greens')
ax.scatter(x5,y5,z5, marker='o', s=15, cmap='Greens')
ax.scatter(x6,y6,z6, marker='o', s=15, cmap='Greens')
plt.show()