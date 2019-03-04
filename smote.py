# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:47:20 2018

@author: snowisland
"""
import matplotlib
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN  # 过抽样处理库SMOTE
from imblearn.under_sampling import RandomUnderSampler  # 欠抽样处理库RandomUnderSampler
from imblearn.ensemble import EasyEnsemble  # 简单集成方法EasyEnsemble
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

from imblearn.combine import SMOTEENN  # 抽样方法

from sklearn.ensemble import RandomForestClassifier  # 随机森

# 导入数据文件
df = pd.read_csv('newtrain.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index("timestamp", inplace=True)
data = df.loc[:, ['value']]  # 切片，得到输入x
label = df.loc[:, ['label']]  # 切片，得到标签y
data.plot(figsize=(10, 3))
df = pd.concat([df, pd.DataFrame(columns=list('ABCDE'))])
df['A'] = data.rolling(window=100).mean()  # 移动窗口的均值
df['B'] = data.rolling(window=100).std()  # 移动窗口的标准差
df['C'] = data.rolling(window=100).median()  # 移动窗口的中位数
df['D'] = data.rolling(window=100).max()  # 移动窗口的最小值
df['E'] = data.rolling(window=100).min()  # 移动窗口的最大值

x = df.iloc[99:, 0:5]  # 切片，得到输入x
y = df.iloc[99:, 6:7]  # 切片，得到标签y

groupby_data_orgianl = df.groupby('label').count()  # 对label做分类汇总
print(groupby_data_orgianl)  # 打印输出原始数据集样本分类分布

model_smote = SMOTEENN()  # 建立SMOTE模型对象
x_smote_resampled_1, y_smote_resampled_1 = model_smote.fit_resample(x, y)  # 输入数据并作过抽样处理
print(x,y)
print('qqqqqqqqqqqqqqqqqqqqqqqqqqqq',x_smote_resampled_1, y_smote_resampled_1 )
x_balance_2 = pd.DataFrame(x_smote_resampled_1, columns=['col1', 'col2', 'col3', 'col4', 'col5'])  # 将数据转换为数据框并命名列名
y_balance_2 = pd.DataFrame(y_smote_resampled_1, columns=['label'])  # 将数据转换为数据框并命名列名
smote_resampled = pd.concat([x_balance_2, y_balance_2], axis=1)  # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('label').count()  # 对label做分类汇总
#print(groupby_data_smote)  # 打印输出经过SMOTE处理后的数据集样本分类分布
x_train_balance_2, x_test_balance_2, y_train_balance_2, y_test_balance_2 = train_test_split(x_balance_2, y_balance_2,
                                                                                            test_size=0.3,
                                                                                            random_state=7)
print(x_train_balance_2, x_test_balance_2, y_train_balance_2, y_test_balance_2)
model_smote = SVC()  # 创建SVC模型对象并指定类别权重
model_smote.fit(x_train_balance_2, y_train_balance_2)  # 输入x和y并训练模型
pre_y_2 = model_smote.predict(x_test_balance_2)
print("2：SMOTE方法异常样本过采样+SVM训练结果：")
print("ACC:", accuracy_score(y_test_balance_2, pre_y_2), "recall:", recall_score(y_test_balance_2, pre_y_2), "f-score:",
      f1_score(y_test_balance_2, pre_y_2))

model_adasyn = ADASYN()  # 建立SMOTE模型对象
x_adasyn_resampled_1, y_adasyn_resampled_1 = model_adasyn.fit_sample(x, y)  # 输入数据并作过抽样处理
x_balance_3 = pd.DataFrame(x_adasyn_resampled_1, columns=['col1', 'col2', 'col3', 'col4', 'col5'])  # 将数据转换为数据框并命名列名
y_balance_3 = pd.DataFrame(y_adasyn_resampled_1, columns=['label'])  # 将数据转换为数据框并命名列名
adasyn_resampled = pd.concat([x_balance_3, y_balance_3], axis=1)  # 按列合并数据框
groupby_data_adasyn = adasyn_resampled.groupby('label').count()  # 对label做分类汇总
print(groupby_data_adasyn)  # 打印输出经过ADASYN处理后的数据集样本分类分布
x_train_balance_3, x_test_balance_3, y_train_balance_3, y_test_balance_3 = train_test_split(x_balance_3, y_balance_3,
                                                                                            test_size=0.3,
                                                                                            random_state=7)
model_adasyn = SVC()  # 创建SVC模型对象并指定类别权重
model_adasyn.fit(x_train_balance_3, y_train_balance_3)  # 输入x和y并训练模型
pre_y_3 = model_adasyn.predict(x_test_balance_3)
print("3:ADASYN方法异常样本过采样+SVM训练结果：")
print("ACC:", accuracy_score(y_test_balance_3, pre_y_3), "recall:", recall_score(y_test_balance_3, pre_y_3), "f-score:",
      f1_score(y_test_balance_3, pre_y_3))

"""
#
#
## 1：原始数据+SVM 不做样本均衡
clf = SVC()
clf.fit(x_train,y_train)
pre_y = clf.predict(x_test)
print("1：原始数据+SVM训练结果：")
print("ACC:",accuracy_score(y_test, pre_y),"recall:",recall_score(y_test, pre_y),"f-score:",f1_score(y_test, pre_y))
#
#
## 2：使用SMOTE方法进行过抽样处理
x_balance_2,y_balance_2 = Sample_Balance(x,y)
x_train_balance_2, x_test_balance_2,  y_train_balance_2, y_test_balance_2 = train_test_split(x_balance_2, y_balance_2, test_size = 0.3, random_state = 7)
model_smote = SVC() # 创建SVC模型对象并指定类别权重
model_smote.fit(x_train_balance_2,y_train_balance_2) # 输入x和y并训练模型
pre_y_2 = clf.predict(x_test_balance_2)
print("2：SMOTE方法异常样本过采样+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_2, pre_y_2),"recall:",recall_score(y_test_balance_2, pre_y_2),"f-score:",f1_score(y_test_balance_2, pre_y_2))
#
## 3：使用RandomUnderSampler方法进行欠抽样处理
x_balance_3,y_balance_3 = Sample_Balance(x,y)
x_train_balance_3, x_test_balance_3,  y_train_balance_3, y_test_balance_3 = train_test_split(x_balance_3, y_balance_3, test_size = 0.3, random_state = 7)
model_smote = SVC() # 创建SVC模型对象并指定类别权重
model_smote.fit(x_train_balance_3,y_train_balance_3) # 输入x和y并训练模型
pre_y_3 = clf.predict(x_test_balance_3)
print("3:RandomUnderSampler方法欠采样+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_3, pre_y_3),"recall:",recall_score(y_test_balance_3, pre_y_3),"f-score:",f1_score(y_test_balance_3, pre_y_3))
#
## 4：使用SVM的权重调节处理不均衡样本
model_svm = SVC(class_weight='balanced') # 创建SVC模型对象并指定类别权重
model_svm.fit(x_train,y_train) # 输入x和y并训练模型
pre_y_4 = clf.predict(x_test)
print("4:原始数据+加权SVM训练结果：")
print("ACC:",accuracy_score(y_test, pre_y_4),"recall:",recall_score(y_test, pre_y_4),"f-score:",f1_score(y_test, pre_y_4))
#
## 5：使用集成方法EasyEnsemble处理不均衡样本
model_EasyEnsemble = EasyEnsemble() # 建立EasyEnsemble模型对象
x_EasyEnsemble_resampled, y_EasyEnsemble_resampled =model_EasyEnsemble.fit_sample(x, y) # 输入数据并应用集成方法处理
print (x_EasyEnsemble_resampled.shape) # 打印输出集成方法处理后的x样本集概况
print (y_EasyEnsemble_resampled.shape) # 打印输出集成方法处理后的y标签集概况
## 抽取其中一份数据做审查
index_num = 1 # 设置抽样样本集索引
x_EasyEnsemble_resampled_t =pd.DataFrame(x_EasyEnsemble_resampled[index_num],columns=['col1','col2','col3','col4','col5'])
## 将数据转换为数据框并命名列名
y_EasyEnsemble_resampled_t =pd.DataFrame(y_EasyEnsemble_resampled[index_num],columns=['label']) # 将数据转换为数据框并命名列名
EasyEnsemble_resampled = pd.concat([x_EasyEnsemble_resampled_t,
y_EasyEnsemble_resampled_t], axis = 1) # 按列合并数据框
groupby_data_EasyEnsemble =EasyEnsemble_resampled.groupby('label').count() # 对label做分类汇总
print (groupby_data_EasyEnsemble) # 打印输出经过EasyEnsemble处理后的数据集样本分类分布
x_train_balance_5, x_test_balance_5,  y_train_balance_5, y_test_balance_5 = train_test_split(x_EasyEnsemble_resampled_t, y_EasyEnsemble_resampled_t, test_size = 0.3, random_state = 7)
model_EasyEnsemble = SVC() # 创建SVC模型对象并指定类别权重
model_EasyEnsemble.fit(x_train_balance_5,y_train_balance_5) # 输入x和y并训练模型
pre_y_5 = clf.predict(x_test_balance_5)
print("5:EasyEnsemble+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_5, pre_y_5),"recall:",recall_score(y_test_balance_5, pre_y_5),"f-score:",f1_score(y_test_balance_5, pre_y_5))
#
#
print("1：原始数据+SVM训练结果：")
print("ACC:",accuracy_score(y_test, pre_y),"recall:",recall_score(y_test, pre_y),"f-score:",f1_score(y_test, pre_y))
print("2：SMOTE方法异常样本过采样+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_2, pre_y_2),"recall:",recall_score(y_test_balance_2, pre_y_2),"f-score:",f1_score(y_test_balance_2, pre_y_2))
print("3:RandomUnderSampler方法欠采样+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_3, pre_y_3),"recall:",recall_score(y_test_balance_3, pre_y_3),"f-score:",f1_score(y_test_balance_3, pre_y_3))
print("4:原始数据+加权SVM训练结果：")
print("ACC:",accuracy_score(y_test, pre_y_4),"recall:",recall_score(y_test, pre_y_4),"f-score:",f1_score(y_test, pre_y_4))
print("5:EasyEnsemble+SVM训练结果：")
print("ACC:",accuracy_score(y_test_balance_5, pre_y_5),"recall:",recall_score(y_test_balance_5, pre_y_5),"f-score:",f1_score(y_test_balance_5, pre_y_5))
#
#"""
