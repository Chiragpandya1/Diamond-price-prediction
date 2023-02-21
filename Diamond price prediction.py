#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# ![diamond-1199183__480.webp](attachment:diamond-1199183__480.webp)

# ![download.jpg](attachment:download.jpg)

# In[6]:


df=pd.read_csv(r"E:\data from keggal\data from keggal\Diamonds Prices2022.csv")
df


# In[7]:


df.drop(['Unnamed: 0'],axis=1 , inplace = True)


# In[8]:


plt.figure(figsize = (15, 10))
sns.heatmap(df.corr(), annot = True,)
plt.show()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[6]:


df[df["x"]==0]
df[df["y"]==0]
df[df["z"]==0]


# In[7]:


df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)
df.shape


# In[8]:


# by this command 20 values has been removed(53943 to 53923)


# In[9]:


dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
print(df.shape)


# In[10]:


print('Before',df.shape)
df.drop_duplicates(inplace=True) 
print('After',df.shape)


# # Outlier check

# In[11]:


cols = ['carat','depth', 'table', 'x', 'y', 'z',
       'price' ]
for i in cols:
    sns.boxplot(df[i],whis=1.5)
    plt.show();


# In[12]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[13]:


for column in df[cols].columns:
    lr,ur=remove_outlier(df[column])
    df[column]=np.where(df[column]>ur,ur,df[column])
    df[column]=np.where(df[column]<lr,lr,df[column])


# In[14]:


cols = ['carat','depth', 'table', 'x', 'y', 'z',
       'price' ]
for i in cols:
    sns.boxplot(df[i],whis=1.5)
    plt.show();


# In[51]:


df.isnull().sum()


# # Visualizations

# In[16]:


sns.barplot('clarity','price',data=df)


# In[17]:


sns.barplot('cut','price',data=df)


# In[18]:


df.groupby(['cut']).sum().plot(kind='pie', y='price')


# In[19]:


df.groupby(['clarity']).sum().plot(
    kind='pie', y='price', autopct='%1.0f%%')


# In[20]:


df.groupby(['color']).sum().plot(
    kind='pie', y='depth', autopct='%1.1f%%')


# In[21]:


sns.pairplot(df, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, 
                 kind='scatter', diag_kind='auto', height=2.5, aspect=1, )


# In[22]:


# data visualization
plt.figure(figsize =[12,12])
plt.subplot(221)
#carat weight distribution
plt.hist(df['carat'],bins = 20,color ='b')
plt.xlabel("carat weight")
plt.ylabel("Frequency")
plt.title("Distribution of diamond carat wieght")
plt.subplot(222)
#distribution of price value
plt.hist(df['price'],bins=20,color='r')
plt.xlabel("Diamond price")
plt.ylabel("Frequency")
plt.title
("Diamond Price distribution")


# # handeling Catagorical Features

# In[52]:


# Label Encoding
df['cut']=df['cut'].map({'Fair':0,'Good':1,'Very Good':2,'Premium':3,'Ideal':4})


# In[53]:


# Dummy Encoding
df_color=pd.get_dummies(df['color'],drop_first=True)
df_color
df_clarity=pd.get_dummies(df['clarity'],drop_first=True)
df_clarity


# In[54]:


# Concat them(df_color and df_clarity)
df=pd.concat([df,df_color,df_clarity,],axis = 1)
df


# In[55]:


df=df.drop(['clarity','color','cut'],axis=1)


# In[27]:


df


# In[56]:


y = df.price
x = df.drop(["price"], axis = 1)


# In[30]:


y


# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.20,random_state=0)


# In[58]:


print(f'length of x train data is {len(x_train)}')
print(f'length of y train data is {len(y_train)}')
print(f'length of x test data is {len(x_test)}')
print(f'length of y test data is {len(y_test)}')


# # Standard scaler

# In[59]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# # linear regression

# In[60]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[61]:


lr.fit(x_train,y_train)


# In[62]:


lr_pred=lr.predict(x_test)


# # Accuracy of LinearRegression Model

# In[63]:


from sklearn.metrics import r2_score
lr=r2_score(y_test,lr_pred)*100 
print(lr)


# In[64]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,lr_pred)


# In[65]:


import math

MSE = np.square(np.subtract(y_test,lr_pred)).mean() 
print(f' mean square ={MSE}')
RMSE = math.sqrt(MSE)
print(f"Root Mean Square Error={RMSE}")


# # Decision Tree Regressor Algorithm

# In[67]:


from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor()
reg.fit(x_train,y_train)
reg_pred=reg.predict(x_test)


# # Accuracy score of Decision Tree Regressor Model

# In[68]:


from sklearn.metrics import r2_score
dtr=r2_score(y_test,reg_pred)*100 
print(dtr)


# In[69]:


import math

MSE = np.square(np.subtract(y_test,reg_pred)).mean() 
print(f' mean square ={MSE}')
RMSE = math.sqrt(MSE)
print(f"Root Mean Square Error={RMSE}")


# # Random Forest Regressor Algorithm

# In[70]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=50)
rf.fit(x_train,y_train)
rf_pred=rf.predict(x_test)


# # Accuracy score of Random Forest Regressor Model

# In[71]:


from sklearn.metrics import r2_score
raf=r2_score(y_test,rf_pred)*100
print(raf)


# In[72]:


import math

MSE = np.square(np.subtract(y_test,rf_pred)).mean() 
print(f' mean square ={MSE}')
RMSE = math.sqrt(MSE)
print(f"Root Mean Square Error={RMSE}")


# # k-Neighbors Regression Algorithm

# In[73]:


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors = 5)
knn.fit(x_train,y_train)
knn_pred=knn.predict(x_test)


# # Accuracy score of Random Forest Regressor Model

# In[74]:


from sklearn.metrics import r2_score
KNN=r2_score(y_test,knn_pred)*100
print(KNN)


# In[75]:


import math

MSE = np.square(np.subtract(y_test,knn_pred)).mean() 
print(f' mean square ={MSE}')
RMSE = math.sqrt(MSE)
print(f"Root Mean Square Error={RMSE}")


# # SVR algorithm

# In[76]:


from sklearn.svm import SVR
svr =SVR()
svr.fit(x_train,y_train)
svr_pred = svr.predict(x_test)


# # Accuracy Score of SVR algorithm

# In[77]:


svrscore = r2_score(y_test,svr_pred)*100
print(svrscore)


# In[78]:


import math

MSE = np.square(np.subtract(y_test,svr_pred)).mean() 
print(f' mean square ={MSE}')
RMSE = math.sqrt(MSE)
print(f"Root Mean Square Error={RMSE}")


# In[ ]:


"""Observation
from the following scores we can conclude that random forest has given best r square and least Rmse
which show that model is perfect,less rmse is good for model.
worst accuracy is shown by SVR """

