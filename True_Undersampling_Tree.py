'''
-*- AI project -*-
Using the decesion tree model to test dataset Financial Distress.csv
The data preprossing method is keeping the last records of
'''
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
df2 = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(df))
# print(df.head)
# print(df.C.unique().shape)
# print(df.describe())
print('df.shape', df.shape)
print('df2.shape', df2.shape)

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


# 欠采样数据处理方法df
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

# Tokenlization
Y = new_fd[:, 2]
for i in range(0, len(Y)):
    if Y[i] > -0.5:
        Y[i] = 0
    else:
        Y[i] = 1

# Preprocessing the data and split the dataset into traning and testing sets
X = new_fd[:, 3:86]
print('X.shape', X.shape)

# delete the eightieth column
pre_X = np.delete(X, 79, axis=1)
print('pre_X.shape', pre_X.shape)


# 标准化
new_fd_scaled = StandardScaler()
new_fd_scaled.fit(pre_X)
new_fd_TrsX = new_fd_scaled.transform(pre_X)

# 划分训练集测试集
# trainX, testX, trainY, testY = train_test_split(new_fd_TrsX, Y, test_size=0.3, random_state=1)


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


# 建树，算法会优先考虑满足max_lead_node,再考虑max_depth,实验目的是找到最高的准确率和f1_score值
dtree = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_features=None, max_depth=9, max_leaf_nodes=10)
dtree.fit(new_fd_TrsX, Y)
# updated dtre.fit(df2_TrsX, Y)
Y_pred = dtree.predict(df2_TrsX)
# print(testX)
# print(Y_pred)
# print("Accuracy: \n", dtree.score(trainX, testY))
print(metrics.accuracy_score(Y_2, Y_pred))
print(f1_score(Y_2, Y_pred))
# print(np.unique(Y, return_counts=True))
# print(np.unique(Y_2, return_counts=True))
# print(np.unique(Y_pred, return_counts=True))


# 画树
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("original4.pdf")




