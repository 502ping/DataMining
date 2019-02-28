'''
修正的欠采样knn
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
df2 = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')

# 未经过采样或欠采样处理的df2
X_2 = df2.values[:, 3:86]
Y_2 = df2.values[:, 2]
for i in range(0, len(Y_2)):
    if Y_2[i] > -0.5:
        Y_2[i] = 0
    else:
        Y_2[i] = 1

# Preprocessing the data and split the dataset into traning and testing sets df2
# X = new_fd[:, 3:86]
# print(X.shape)
# X_2 = df2[:, 3:86]
# print(X_2.shape)

# delete the eightieth column  df2
pre_X_2 = np.delete(X_2, 79, axis=1)
print('pre_X_2.shape', pre_X_2.shape)


# 标准化df2
df2_scaled = StandardScaler()
df2_scaled.fit(pre_X_2)
df2_TrsX = df2_scaled.transform(pre_X_2)

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
# print(type(new_fd), "\n", new_fd.shape, "\n", new_fd)


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
print('pre_X', pre_X.shape)


# 标准化
new_fd_scaled = StandardScaler()
new_fd_scaled.fit(pre_X)
new_fd_TrsX = new_fd_scaled.transform(pre_X)

'''
这里没有必要做数据集的划分
# 划分训练集测试集
# trainX, testX, trainY, testY = train_test_split(new_fd_TrsX, Y, test_size=0.3, random_state=1)

'''

'''
# 重复检测
count = 0
frame = pd.DataFrame(new_fd)
print(frame.shape)
IsDuplicated = frame.duplicated()
print(IsDuplicated)
frame = frame.drop_duplicated(['state'])
print(frame.shape)

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
'''


# knn模型
neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', p=2, metric='euclidean')
# k= 1是最高的
neigh.fit(new_fd_TrsX, Y)
Y_pred = neigh.predict(df2_TrsX)

# neigh.predict_proba(TrsX)
# print(np.unique(Y, return_counts=True))
# print(np.unique(Y_2, return_counts=True))
# print(np.unique(Y_pred, return_counts=True))

print(metrics.accuracy_score(Y_2, Y_pred))
print(f1_score(Y_2, Y_pred))
