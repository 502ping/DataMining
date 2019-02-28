'''
原始的KNN，数据集没有经过欠采样和过采样的处理，
仍然划分训练集和测试集，且仅使用f1_score为模型的衡量
一般来讲针对不平衡的数据集，knn参数k值采用1得到的f1_score最大
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(new_fd))
# print(new_fd.C.unique().shape)
# print(new_fd.describe())

# 预处理Y
Y = df.values[:, 2]
for i in range(0, len(Y)):
    if Y[i] > -0.5:
        Y[i] = 0
    else:
        Y[i] = 1
print(Y.shape)

# Preprocessing the data and split the dataset into traning and testing sets
X = df.values[:, 3:86]
print(X.shape)
# 删除第80行
pre_X = np.delete(X, 79, axis=1)
print(pre_X.shape)


# 标准化
df_scaled = StandardScaler()
df_scaled.fit(pre_X)
df_TrsX = df_scaled.transform(pre_X)

# 划分训练集测试集
trainX, testX, trainY, testY = train_test_split(df_TrsX, Y, test_size=0.3, random_state=1)


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


# knn模型
# 在不平衡的数据中，K = 1 的时候f1_score应该最高
neigh = KNeighborsClassifier(n_neighbors=1, weights='distance', algorithm='auto', p=2, metric='euclidean')
neigh.fit(trainX, trainY)
Y_pred = neigh.predict(testX)
neigh.predict(testX)
neigh.predict_proba(testX)
# 不平衡的原始数据不用准确率参考
# print(metrics.accuracy_score(testY, Y_pred))
print(f1_score(testY, Y_pred))
