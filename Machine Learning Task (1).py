#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as p
import numpy as n
import matplotlib.pyplot as m
import seaborn as s

data = p.read_csv(r"C:\Users\annar\Downloads\data.csv\data.csv")
data


# In[7]:


data = data.drop("date", axis=1)
data


# In[8]:


data.describe()


# In[9]:


data.shape


# In[10]:


data.corr()


# In[11]:


correlation = data.corr()
m.figure(figsize=(10,10))
s.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')


# In[12]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm


# In[13]:


# assigning data to variables
y = data['number_people'].values
data = data.drop(['number_people','is_weekend'],axis=1)
X = data.values
data.head()


# In[14]:


# splitting data into test and train data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01)
# fitting the data and transforming
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# Goal 1- Given a time of day (and maybe some other features, including weather), predict how crowded the gym will be.

# In[22]:


from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(Xtrain,ytrain)
print(linreg.score(Xtest,ytest)*100)
print(linreg.score(Xtrain,ytrain)*100)


# In[15]:


# Using RandomForestRegressor() for making the prediction
radm = RandomForestRegressor()
radm.fit(Xtrain, ytrain)
y_val_l = radm.predict(Xtest)
print("Result: ",radm.score(Xtest, ytest)*100)


# Goal 2- Features which are most important.

# In[20]:


indices = n.argsort(radm.feature_importances_)[::-1]
print('Feature ranking:')
for f in range(data.shape[1]):
    print('%d. feature %d %s (%f)' % (f+1 , indices[f], data.columns[indices[f]],radm.feature_importances_[indices[f]]*100))


# In[ ]:




