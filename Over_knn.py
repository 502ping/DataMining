'''
-*- AI project -*-
Using knn model to test dataset Financial Distress.csv
The data prepocessing method is 10 times duplicating the imbalanced date
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(df))
# print(df.C.unique().shape)
# print(df.describe())

# Preprocessing the data and split the dataset into traning and testing sets
X = df.values[:, 3:86]
Y = df.values[:, 2]


# 初始化两个空集来接收数据
distress_dataset = np.zeros(shape=(136, 86))
healthy_dataset = np.zeros(shape=(3536, 86))
distress_count = 0
healthy_count = 0

for i in range(0, len(Y)):
    if Y[i] > -0.5:
        healthy_dataset[healthy_count] = df.values[i]
        healthy_count = healthy_count + 1
    else:
        distress_dataset[distress_count] = df.values[i]
        distress_count = distress_count + 1

print("Healthy", healthy_dataset.shape)
print("Distress", distress_dataset.shape)
guo_dataset = np.vstack((distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset, distress_dataset))
guo = np.vstack((guo_dataset, healthy_dataset))
print("Preprocessed", guo.shape)
X = guo[:, 3:86]
Y = guo[:, 2]

for i in range(0, len(Y)):
    if Y[i] > -0.5:
        Y[i] = 0
    else:
        Y[i] = 1
print(np.unique(Y, return_counts=True))


# 标准化
df_scaled = StandardScaler()
df_scaled.fit(X)
df_TrsX = df_scaled.transform(X)

# 划分训练集测试集
trainX, testX, trainY, testY = train_test_split(df_TrsX, Y, test_size=0.3, random_state=1)


# 重复检测
'''
count = 0
frame = pd.DataFrame(df)
print(frame.shape)
IsDuplicated = frame.duplicated()
print(IsDuplicated)
frame = frame.drop_duplicated(['state'])
print(frame.shape)
'''


# KNN模型
neigh = KNeighborsClassifier(n_neighbors=11, weights='distance', algorithm='auto', p=2, metric='euclidean')
neigh.fit(trainX, trainY.astype('int'))
Y_pred = neigh.predict(testX)
neigh.predict(testX)
neigh.predict_proba(testX)
print(metrics.accuracy_score(testY, Y_pred))
print(f1_score(testY, Y_pred))
print(np.unique(Y_pred, return_counts=True))
print(neigh.kneighbors_graph())
