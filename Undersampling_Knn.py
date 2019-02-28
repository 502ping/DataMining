# -*- 2018.12.08 By. xyc -*-
# -*- AI project -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(new_fd))
# print(new_fd.C.unique().shape)
# print(new_fd.describe())

# 欠采样数据处理方法
rows, columns = df.shape
new_fd = np.zeros(shape=(422, 86))
n = 0
for i in range(rows-1):
    if i+1 == rows-1:
        new_fd[n] = df.values[i+1]
        n = n + 1
    if df.values[i, 0] != df.values[i+1, 0]:
        new_fd[n] = df.values[i]
        n = n + 1
print(type(new_fd), "\n", new_fd.shape, "\n", new_fd)


# 预处理Y
Y = new_fd[:, 2]
for i in range(0, len(Y)):
    if Y[i] > -0.5:
        Y[i] = 0
    else:
        Y[i] = 1


# Preprocessing the data and split the dataset into traning and testing sets
X = new_fd[:, 3:86]
print(X.shape)
# 删除第80行
pre_X = np.delete(X, 79, axis=1)
print(pre_X.shape)


# 标准化
new_fd_scaled = StandardScaler()
new_fd_scaled.fit(pre_X)
new_fd_TrsX = new_fd_scaled.transform(pre_X)

# 划分训练集测试集
trainX, testX, trainY, testY = train_test_split(new_fd_TrsX, Y, test_size=0.3, random_state=1)


'''
# 重复检测
count = 0
frame = pd.DataFrame(new_fd)
print(frame.shape)
IsDuplicated = frame.duplicated()
print(IsDuplicated)
frame = frame.drop_duplicated(['state'])
print(frame.shape)
'''

# 缺失查找
# missing = new_fd.columns[new_fd.isnull().any()].tolist()
# new_fd[missing].isnull().sum()
# if missing == []:
#     print("No missing data.")
# else:
#     print(missing + new_fd[missing].isnull().sum())


# 验证不平衡性
# scaler = StandardScaler()
# trainArray = new_fd.as_matrix()
# scaledData = trainArray
# scaledData[:, 1:] = scaler.fit_transform(trainArray[:, 1:])
# print(np.sum(scaledData[:, 84] > -0.5))
# print(np.sum(scaledData[:, 84] <= -0.5))


# knn模型
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', p=2, metric='euclidean')
# k= 5/7 都可以
neigh.fit(trainX, trainY)
Y_pred = neigh.predict(testX)
neigh.predict(testX)
neigh.predict_proba(testX)
print(metrics.accuracy_score(testY, Y_pred))
print(f1_score(testY, Y_pred))
