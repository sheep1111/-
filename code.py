import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalizing(x):#将特征进行归一化处理
    m,n = x.shape
    for j in range(n):
        features = x[:,j]
        min = features.min(axis = 0)
        max = features.max(axis = 0)
        differ = max - min
        if differ != 0 :
            x[:,j] = (features - min)/differ
        else:
            x[:,j] = 0
    #print(x)
    return x

def cost (theta,X,Y):#损失函数
    diff = np.dot(X,theta)-Y#347 X 1
    return (1/(2*m))*np.dot(diff.T,diff)#1 X 1

def gradient(theta,X,Y):#偏导函数
    m = len(X)  # 统计样本个数
    diff = np.dot(X,theta) - Y#347 X 1
    return (1/m)*np.dot(X.T,diff)#12 X 1

def descent(X,Y,alpha):#梯度下降
    theta = np.random.randint(-1, 1, size=(x.shape[1], 1))  # 随机生成theta值
    grad = gradient(theta,X,Y)
    i = 0
    I = []
    C = []
    V = 1
    for k in range(20000):
        i+=1
        theta = theta*(1 - (alpha*V)/m) - alpha * grad#添加正则化项防止过拟合
        grad = gradient(theta,X,Y)
        cost_ = cost (theta,X,Y)
        C.append(cost_)
        I.append(i)

    plt.scatter(I, C)
    plt.xlabel("iterations")  # x轴上的名字
    plt.ylabel("loss")  # y轴上的名字
    plt.show()

    print(i)
    return theta

df=pd.read_csv('train_.csv')#train数据的导入
#df.drop_duplicates()#去除重复的数据
data = np.array(df,type(float))#将全部的数据用阵形式表现
np.random.shuffle(data)#打乱原有数据集的顺序
#将nan值用本特征的平均值代替
col_mean = np.nanmean(data,axis=0)#计算均值
inds = np.where(pd.isnull(data))#找到nan的索引
data[inds] = np.take(col_mean,inds[1])#替换

#print(col_mean)
x =data[:,1:14]#每个样本有13个分量

x = normalizing(x)#将样本数据进行归一           #404 X 13
Z = np.column_stack((x,x**2))
x = np.c_[np.ones(x.shape[0]), Z]#x0为1     #404 X 14
y=data[:,14:15]#对应的y的输出                 #404 X 1

theta = np.random.randint(-1, 1, size=(x.shape[1]+1, 1))  # 随机生成theta值
#df.dropna(axis=0, how='any', inplace=True)#数据处理删掉nan的数          #14 X 1
m = len (x)#统计样本个数            #404
#print(m)

alpha = 0.03

Theta = descent(x,y,alpha)
L = cost(Theta,x,y)
print(L)
#利用正则化测试比较L和L2
x=np.array(x, dtype=float)
inverse = np.linalg.inv(np.dot(x.T,x))
theta2 = np.dot(inverse,np.dot(x.T,y))
L2 = cost(theta2,x,y)
print(L2)
#print(Theta)

#test
df=pd.read_csv('test_.csv')
data_2 = np.array(df,type(float))#将全部的数据用矩阵形式表现

#将nan值用本特征的平均值代替
col_mean = np.nanmean(data_2,axis=0)#计算均值
inds = np.where(pd.isnull(data_2))#找到nan的索引
data_2[inds] = np.take(col_mean,inds[1])#替换

x_data2=data_2[:,1:14]#每个样本有13个分量
X = normalizing(x_data2)#将样本数据进行归一
H = np.column_stack((X,X**2))
#n = len (x_data2)#统计样本个数
b  = np.ones(102)
X = np.c_[b,H]
Y = np.dot(X,Theta)
Y = np.array(Y)
#print(Y)
Y_1 = pd.DataFrame(Y)
Y_1.to_csv('out_.csv')