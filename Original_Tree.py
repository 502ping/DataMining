# -*- 2018.12.07 By. xyc -*-
# -*- AI project -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(df))
# print(df.C.unique().shape)
# print(df.describe())

# Preprocessing the data and split the dataset into traning and testing sets
X = df.values[:, 3:86]
Y = df.values[:, 2]
for i in range(0, len(Y)):
    if Y[i] > -0.5:
        Y[i] = 0
    else:
        Y[i] = 1
# print(Y.shape, np.count_nonzero(Y))

# delete the x80 column
pre_X = np.delete(X, 79, axis=1)
print(pre_X.shape)

# 标准化
df_scaled = StandardScaler()
df_scaled.fit(X)
df_TrsX = df_scaled.transform(X)

# 划分训练集测试集
trainX, testX, trainY, testY = train_test_split(df_TrsX, Y, test_size=0.3, random_state=1)


# Check the duplicating value
# frame = pd.DataFrame(df)
# print(frame.shape)
# IsDuplicated = frame.duplicated()
# print(IsDuplicated)
# frame = frame.drop_duplicated(['state'])
# print(frame.shape)


# Check the vacancy
missing = df.columns[df.isnull().any()].tolist()
df[missing].isnull().sum()
if missing == []:
    print("No missing data.")
else:
    print("missing", df[missing].isnull().sum())

# 处理偏离值

# 验证不平衡性
# scaler = StandardScaler()
# trainArray = df.as_matrix()
# scaledData = trainArray
# scaledData[:, 1:] = scaler.fit_transform(trainArray[:, 1:])
# print(np.sum(scaledData[:, 84] > -0.5))
# print(np.sum(scaledData[:, 84] <= -0.5))s


# 建树
dtree = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_features=None, max_leaf_nodes=24, class_weight="balanced")
# max_depth = 8 剪枝，树会简单一点
dtree.fit(trainX, trainY)
Y_pred = dtree.predict(testX)
# print(testX)
# print(Y_pred)
# print("Accuracy: \n", dtree.score(trainX, testY))
print(metrics.accuracy_score(testY, Y_pred))
print(f1_score(testY, Y_pred))


# # 画树
# dot_data = StringIO()
# export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("tree_original_11_41.pdf")




