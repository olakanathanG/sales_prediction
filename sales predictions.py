#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


dataset=pd.read_csv("F:\svm dataset.csv")
dataset


# In[16]:


import matplotlib.pyplot as plt
plt.scatter(dataset.sales,dataset.price,color='blue',marker='*')
plt.show


# In[17]:


x=dataset.iloc[:,:-1]
x


# In[23]:


y=dataset.iloc[:,-1]
y


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[26]:


from sklearn.linear_model import LinearRegression
RG=LinearRegression()
RG.fit(x_train,y_train)


# In[28]:


rsquare=RG.score(x_test,y_test)
print(rsquare)


# In[34]:


n=len(dataset)
p=len(dataset.columns)-1
twosquare=1-(1-rsquare)*(n-1)/(n-p-1)
print(twosquare)
plt.scatter(rsquare,twosquare,color="blue",marker="*")
plt.show


# In[33]:


x=5644
pere=[[x]]
model=RG.predict(pere)
print(model)


# In[ ]:




