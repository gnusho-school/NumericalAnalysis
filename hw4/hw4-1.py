import numpy as np

point_x=[-2.9,-2.1,-0.9,1.1,0.1,1.9,3.1,4.0]
point_y=[35.4,19.7,5.7,2.1,1.2,8.7,25.7,41.5]

def make_Ab(not1,not2):
	global point_x,point_y
	tmp_A=[]
	tmp_b=[]

	for i in range(8):
		if i==not1:continue
		elif i==not2:continue
		a=point_x[i]
		list_=[a*a,a,1]
		tmp_A.append(list_)
		tmp_b.append(point_y[i])

	A=np.array(tmp_A)
	b=np.array(tmp_b)

	return A,b

def cal_p(A,b):
	p=np.linalg.inv(A.T@A)@A.T@b
	return p

def main():
	A1,b1=make_Ab(0,4)
	p1=cal_p(A1,b1)
	print(p1)
	A2,b2=make_Ab(2,6)
	p2=cal_p(A2,b2)
	print(p2)

if __name__=="__main__":
	main()

#2차원 곡선에서 약간씩 어긋난 좌표
#보고서에 쓸 내용
#=> 2개의 결과 비교 및 sample data에 따라서 값이 달라지는 것을 확인 (장단점?)
#=> 파워포인트에 간단하게 행렬을 가지고 연산한 과정을 파워포인트에 간단하게 써주면 된다. 