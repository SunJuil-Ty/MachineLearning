import numpy as np
import matplotlib.pyplot as plt
import random

#高斯函数的均值
mean_positive = np.array([1,1])
mean_negtive_1 = np.array([0,0])
mean_negtive_2 = np.array([1,0])
mean_negtive_3 = np.array([0,1])
#高斯函数的协方差矩阵
cov = np.array([[0.01,0],[0,0.01]])
#正类和负类的样本点
f1 = np.random.multivariate_normal(mean_positive,cov,300)
f2 = np.random.multivariate_normal(mean_negtive_1,cov,300)
f3 = np.random.multivariate_normal(mean_negtive_2,cov,300)
f4 = np.random.multivariate_normal(mean_negtive_3,cov,300)
#添加标签
f1_pisitive = f1.tolist()
f2_negtive = f2.tolist()
f3_negtive = f3.tolist()
f4_negtive = f4.tolist()
[i.append(1) for i in f1_pisitive]
[i.append(-1) for i in f2_negtive]
[i.append(-1) for i in f3_negtive]
[i.append(-1) for i in f4_negtive]

trainData = f1_pisitive + f2_negtive + f3_negtive + f4_negtive  #训练集
random.shuffle(trainData)
print(trainData)
alpha = 0.01 #学习参数

def sigmoid(x1:float, x2:float, w1:float, w2:float, b:float):
    y = (w1*x1) + (w2*x2) + b
    if y >=0:
        return 1
    else:
        return -1

def PLA():
    w1 = 0.0
    w2 = 0.0
    b = -0.5
    flag = True
    #k = 0
    while flag == True:
        t = 0   #阈值
        flag = False
        for i in range(len(trainData)):
            pre_label = sigmoid(trainData[i][0],trainData[i][1],w1,w2,b)
            if(pre_label != trainData[i][2]):
                w1 += alpha*trainData[i][2]*trainData[i][0]
                w2 += alpha*trainData[i][2]*trainData[i][0]
                b +=  alpha*trainData[i][2]
                t += 1
                #k += 1
                #print(k)
                flag = True
        if t < 10:
            break
    return w1,w2,b

w1,w2,b = PLA()
x1 = np.linspace(-0, 1.4, 100)
x2 = (-w1/w2)*x1 - b/w2
plt.scatter(f1[:,0],f1[:,1],c='r', marker='+',label='+类')
plt.scatter(f2[:,0],f2[:,1],c='g', marker='.',label='-类')
plt.scatter(f3[:,0],f3[:,1],c='g', marker='.',label='-类')
plt.scatter(f4[:,0],f4[:,1],c='g', marker='.',label='-类')
plt.plot(x1,x2,'-')
plt.xlim(-0.3,1.4)
plt.ylim(-0.3,1.4)
plt.ylabel('x2',fontsize='large')
plt.xlabel('X1',fontsize='large')
plt.annotate('w1:%0.4f'%w1,xy=(0.5,0.6))
plt.annotate('w2:%0.4f'%w2,xy=(0.5,0.5))
plt.annotate('b:%0.4f'%b,xy=(0.5,0.4))
plt.show()