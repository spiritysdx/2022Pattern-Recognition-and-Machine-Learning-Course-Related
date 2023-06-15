#!/usr/bin/env python
# coding: utf-8

# In[16]:


# import numpy as np
# import pandas as pd
# import re,math
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import scipy.io as scio

# 读取数据
# data = scio.loadmat("breast_cancer.mat")
# labels = data['labels'][0]
# data1 = data['data']
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print("size of the data", X.shape, y.shape)


# In[17]:


#标签名称查看
print("target_names:", cancer.target_names)

#特征数量和名称查看
print('feature_names:', cancer.feature_names, len(cancer.feature_names))

#查看阳性和阴性样本
print("Num of two classes:", y[y==0].shape, y[y==1].shape)


# In[18]:


# 数据集划分
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=666,shuffle=True)


# In[19]:


# 支持向量机的分类器的声明
model = svm.SVC(kernel='poly', degree=2)
clf = model.fit(X_train, y_train)

# 测试并输出结果
print(clf.score(X_test, y_test))
print(clf.predict(X_test))

y_score = model.decision_function(X_test)
print(y_score)


# In[ ]:




