#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# 读取数据
cancer = load_breast_cancer()
X,y = cancer.data, cancer.target
print("size of the data", X.shape, y.shape)


# In[53]:


#标准化 x
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)


# In[54]:


# 数据集划分
X_train,X_test,y_train,y_test = train_test_split(X_scaler,y,test_size=0.2,random_state=666,shuffle=True)


# In[55]:


# 声明多层感知器网络
model = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1e-5, 
                      batch_size='auto', beta_1=0.9, beta_2=0.999,
                      epsilon=1e-08, hidden_layer_sizes=(10,5),
                      learning_rate="constant", learning_rate_init=0.001,
                      max_iter=200,momentum=0.9)


# In[56]:


# 训练与测试 MLP
clf = model.fit(X_train, y_train)
print(clf.score(X_test, y_test))

scores = cross_val_score(model, X_scaler, y, cv = 5)
print(scores)


# In[ ]:





# In[ ]:




