import numpy as np
import matplotlib.pyplot as plt
import random
import math

#高斯函数的均值
mean_positive = np.array([1,1])
mean_negtive_1 = np.array([0,0])
mean_negtive_2 = np.array([1,0])
mean_negtive_3 = np.array([0,1])
#高斯函数的协方差矩阵
cov = np.array([[0.002,0],[0,0.002]])
#正类和负类的样本点
f1 = np.random.multivariate_normal(mean_positive,cov,200)
f2 = np.random.multivariate_normal(mean_negtive_1,cov,200)
f3 = np.random.multivariate_normal(mean_negtive_2,cov,200)
f4 = np.random.multivariate_normal(mean_negtive_3,cov,200)
#添加标签
f1_negtive = f1.tolist()
f2_negtive = f2.tolist()
f3_pisitive = f3.tolist()
f4_pisitive = f4.tolist()
[i.append(0) for i in f1_negtive]
[i.append(0) for i in f2_negtive]
[i.append(1) for i in f3_pisitive]
[i.append(1) for i in f4_pisitive]

trainData = f1_negtive + f2_negtive + f3_pisitive + f4_pisitive  #训练集
# random.shuffle(trainData)
# print(trainData)


def putin(x1:float, x2:float, w1:float, w2:float, b:float):
    return x1*w1 + x2*w2 + b

def putout(y:float):
    return 1.0/(1.0+math.e**(-y))

#a = putout((putin(1,1,1,1,-1)))
#print(a)

def grad(x:float):
    return putout(x)*(1.0-putout(x))

def init(n:int):
    return np.random.rand(n)

def BP():
    w1,w2,w3,w4,w5,w6 = init(6)
    b1,b2,b3 = -init(3)
    #print(b1,b2,b3)
    n = 0.1
    t = 0 
    while True:
        E = 0.0
        for i in range(len(trainData)):
            x1,x2,z = trainData[i][:]
            y1_in = putin(x1,x2,w1,w3,b1)
            y1_out = putout(y1_in)
            y2_in = putin(x1,x2,w2,w4,b2)
            y2_out = putout(y2_in)
            z_in = putin(y1_out,y2_out,w5,w6,b3)
            z_out = putout(z_in)
            
            t = t+1
            E = E + 1.0/2 * (z - z_out)**2
            delta_w1 = -(z - z_out) * grad(z_in) * w5 * grad(y1_in) * x1
            delta_w2 = -(z - z_out) * grad(z_in) * w6 * grad(y2_in) * x1
            delta_w3 = -(z - z_out) * grad(z_in) * w5 * grad(y1_in) * x2
            delta_w4 = -(z - z_out) * grad(z_in) * w6 * grad(y2_in) * x2
            delta_b1 = -(z - z_out) * grad(z_in) * w5 * grad(y1_in)
            delta_b2 = -(z - z_out) * grad(z_in) * w6 * grad(y2_in)
            delta_w5 = -(z - z_out) * grad(z_in) * y1_in
            delta_w6 = -(z - z_out) * grad(z_in) * y2_in
            delta_b3 = -(z - z_out) * grad(z_in)

            w1 = w1 - n*delta_w1
            w2 = w2 - n*delta_w2
            w3 = w3 - n*delta_w3
            w4 = w4 - n*delta_w4
            w5 = w5 - n*delta_w5
            w6 = w6 - n*delta_w6
            b1 = b1 - n*delta_b1
            b2 = b2 - n*delta_b2
            b3 = b3 - n*delta_b3
        print(E/800.0)
        print(t)
        if E/800.0 < 0.001:
            break
    return w1,w2,w3,w4,w5,w6,b1,b2,b3

w1,w2,w3,w4,w5,w6,b1,b2,b3 = BP()
x = np.linspace(-0.3, 1.4, 100)
y1 = (-w1/w3)*x - b1/w3
y2 = (-w2/w4)*x - b2/w4
plt.scatter(f1[:,0],f1[:,1],c='r', marker='.',label='-类')
plt.scatter(f2[:,0],f2[:,1],c='r', marker='.',label='-类')
plt.scatter(f3[:,0],f3[:,1],c='g', marker='+',label='+类')
plt.scatter(f4[:,0],f4[:,1],c='g', marker='+',label='+类')
plt.plot(x,y1,'-')
plt.plot(x,y2,'-')
plt.xlim(-0.3,1.4)
plt.ylim(-0.3,1.4)
plt.ylabel('x2',fontsize='large')
plt.xlabel('X1',fontsize='large')
plt.annotate('w1:%0.4f'%w1,xy=(0.5,0.0))
plt.annotate('w2:%0.4f'%w2,xy=(0.5,0.1))
plt.annotate('w3:%0.4f'%w3,xy=(0.5,0.2))
plt.annotate('w4:%0.4f'%w4,xy=(0.5,0.3))
plt.annotate('w5:%0.4f'%w5,xy=(0.5,0.4))
plt.annotate('w6:%0.4f'%w6,xy=(0.5,0.5))
plt.annotate('b1:%0.4f'%b1,xy=(0.5,0.6))
plt.annotate('b2:%0.4f'%b2,xy=(0.5,0.7))
plt.annotate('b3:%0.4f'%b3,xy=(0.5,0.8))
plt.show()