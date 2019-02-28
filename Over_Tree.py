'''
Model: decision tree
Preprocess method: ten times duplicate the imbalanced part of data
'''
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.preprocessing import StandardScaler


# Reading dataset and setting column
df = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
df2 = pd.read_csv('C:/Users/Bill/PycharmProjects/AIproject/Financial Distress.csv')
# print(type(df))
# print(df.C.unique().shape)
# print(df.describe())

# 未经过过采样和欠采样的数据集df2
X_2 = df2.values[:, 3:86]
Y_2 = df2.values[:, 2]
for i in range(0, len(Y_2)):
    if Y_2[i] > -0.5:
        Y_2[i] = 0
    else:
        Y_2[i] = 1

# delete the eightieth column  df2
pre_X_2 = np.delete(X_2, 79, axis=1)
print('pre_X_2.shape', pre_X_2.shape)


# 标准化df2
df2_scaled = StandardScaler()
df2_scaled.fit(pre_X_2)
df2_TrsX = df2_scaled.transform(pre_X_2)

# Preprocessing the data and split the dataset into traning and testing sets
X = df.values[:, 3:86]
Y = df.values[:, 2]

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

# 删除第80列
pre_X = np.delete(X, 79, axis=1)
print("pre_X", pre_X.shape)

# 标准化
df_scaled = StandardScaler()
df_scaled.fit(pre_X)
df_TrsX = df_scaled.transform(pre_X)

# 没有必要划分训练集测试集
# trainX, testX, trainY, testY = train_test_split(df_TrsX, Y, test_size=0.3, random_state=1)


# 建树
dtree = tree.DecisionTreeClassifier(criterion='entropy', random_state=1, max_features=None, max_depth=10, max_leaf_nodes=66)
#10 66
dtree.fit(df_TrsX, Y)
Y_pred = dtree.predict(df2_TrsX)
print(metrics.accuracy_score(Y_2, Y_pred))
print(f1_score(Y_2, Y_pred))


# 画树
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("over_tree_01.pdf")
