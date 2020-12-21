#!/usr/bin/env python
# coding: utf-8

# KNN - Predict Autism

# In[1]:


import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split # training data
from sklearn.preprocessing import StandardScaler # normalização
from sklearn.neighbors import KNeighborsClassifier # model
from sklearn.metrics import confusion_matrix # teste
from sklearn.metrics import f1_score # teste
from sklearn.metrics import accuracy_score # teste


# In[2]:


dataset = pd.read_csv('Final_dataset.csv')


# In[3]:


X = dataset.iloc[:, 0:1259] # todas as variáveis
Y = dataset.iloc[:, 1259] # classifier
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=0, test_size=0.2)


# In[4]:


classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')
classifier.fit(X_train, Y_train)


# In[5]:


y_pred = classifier.predict(X_test)


# In[6]:


cm = confusion_matrix(Y_test, y_pred)
print(cm)


# In[7]:


print(accuracy_score(Y_test, y_pred))

