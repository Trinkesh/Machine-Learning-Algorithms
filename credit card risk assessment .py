#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np


# In[45]:


credit_data = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Credit_Card-Risk-assessment/master/Credit_default_dataset.csv")
credit_data.head()


# In[46]:


credit_data.drop(["ID"],axis = 1,inplace = True)


# In[47]:


credit_data.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'default.payment.next.month']


# In[48]:


credit_data.head()


# In[49]:


credit_data.EDUCATION.value_counts()


# In[50]:


#we can see that there number represents data into levels from school to univercity level we can fillter out the data 
#there to get better results


# In[51]:


credit_data["EDUCATION"] = credit_data["EDUCATION"].map({0:4,1:1,2:2,3:3,5:4,6:4})


# In[52]:


credit_data.EDUCATION.value_counts()


# In[53]:


credit_data.MARRIAGE.value_counts()
#we can combine both 0 and 3 values in between


# In[54]:


credit_data["MARRIAGE"] = credit_data["MARRIAGE"].map({2:2,1:1,0:3})


# In[ ]:





# In[55]:


credit_data.MARRIAGE.value_counts() 


# In[56]:


from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X = credit_data.drop(["default.payment.next.month"],axis = 1)
X = scaler.fit_transform(X)


# In[57]:


X


# In[58]:


y = credit_data["default.payment.next.month"]
y


# In[59]:


#before Xgboost we can have some parameter defining

params = {
    "learning_rate" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth" : [3,4,5,6,8,10,12,15],
    "min_child_weight" :[1,3,5,7],
    "gamma":  [0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.7]
}


# In[60]:


from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


# In[61]:


import xgboost


# In[62]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[63]:


classifier = xgboost.XGBClassifier()


# In[64]:


random_search = RandomizedSearchCV(classifier,param_distributions = params,n_iter=10,scoring="roc_auc",n_jobs=-1,cv = 5,verbose = 3)


# In[65]:


from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y)
timer(start_time)


# In[66]:


random_search.best_estimator_


# In[67]:


random_search.best_params_


# In[68]:


classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.4, gamma=0.1, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[69]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier,X,y,cv = 10)


# In[70]:


score


# In[71]:


score.mean()


# In[ ]:




