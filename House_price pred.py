#!/usr/bin/env python
# coding: utf-8

# In[185]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[186]:


df = pd.read_csv("OneDrive/Desktop/Data/train_data_h.csv")
df.head()


# In[187]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[188]:


## Fill Missing Values

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
df.drop(['Alley'],axis=1,inplace=True)
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)


# In[189]:


df.shape


# In[190]:


df.drop(['Id'],axis=1,inplace=True)


# In[191]:


df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])


# In[192]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[193]:


df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='YlGnBu')
df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df.dropna(inplace=True)


# In[194]:


columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType',
        'SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir',
         'Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']


# In[195]:


def category_onehot_multcols(multcolumns):

    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[196]:


main_df=df.copy()


# In[197]:


test_df = pd.read_csv("OneDrive/Desktop/Data/formtest.csv")


# In[ ]:





# In[198]:


final_df=pd.concat([df,test_df],axis=0)


# In[199]:


final_df=category_onehot_multcols(columns)


# In[200]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[201]:


final_df.head()


# In[ ]:





# In[202]:


df_Train=final_df.iloc[:1422,:]


# In[205]:


df_Train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[206]:


df_Test.shape


# In[207]:


df_Train['SalePrice']


# In[208]:


x=df_Train.drop(['SalePrice'],axis=1)
y=df_Train['SalePrice']


# In[209]:


## XGB regressor


# In[210]:


import xgboost
from sklearn.model_selection import train_test_split
classifier=xgboost.XGBRegressor()
from sklearn.metrics import mean_squared_log_error
X_train,X_test, Y_train,Y_test = train_test_split(x,y)


# In[211]:


classifier.fit(X_train,Y_train)
y_pred = classifier.predict(X_test)
mean_squared_log_error(y_pred,Y_test)


# In[212]:


## Random forrest


# In[213]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train,Y_train)


# In[214]:


y_pred2 = regr.predict(X_test)
mean_squared_log_error(y_pred2,Y_test)


# In[215]:


## Lasso regression


# In[216]:


from sklearn.linear_model import Lasso


# In[217]:


l = Lasso()


# In[221]:


l.fit(X_train,Y_train)
y_pred3= l.predict(X_test)


# In[219]:


mean_squared_log_error(y_pred3,Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




