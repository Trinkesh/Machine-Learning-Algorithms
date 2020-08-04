

import numpy as np


# In[24]:


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


data = pd.read_csv(r"C:\Users\trinkesh\Documents\Churn_Modelling.csv")
data.head()


# In[26]:


X = data.iloc[:,3:13]
y = data.iloc[:,13]


# In[5]:


X


# In[6]:


y


# #we can say that we have independent and non independent varibles
# 
# Creating dummy variables
# Machine learning models cannot process words, for this reason, columns that show categories need to be converted into dummy variables.
# For example, the variable "Gender" will be transformed into a dummy variable with 0 being a female and 1 being a male. However, variables that
# include more than two categories, will be split into more columns.
# For instance, the "Geography" category has three levels, namely Germany, Spain, and France. This variable needs to be transformed into two binary columns.

# In[7]:


cat = [1,2]
cat_col = pd.get_dummies(X.iloc[:,cat],drop_first=True)


# In[8]:


cat_col


# In[9]:


X.drop(X.columns[cat],axis = 1,inplace  =  True)


# In[10]:


X.head()


# In[11]:


X = pd.concat([X,cat_col], axis=1)


# In[12]:


X.head()


# In[13]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train,y_test  = train_test_split(X,y,test_size = 0.2)


# In[14]:


#okay before we go further lets satndardize rhe data
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()


# In[15]:


X_train = pd.DataFrame(standard.fit_transform(X_train),columns = X_train.columns)

X_test = pd.DataFrame(standard.fit_transform(X_test),columns =X_test.columns)
#converting data into standardized formate is vary essential for having to compute data smilerly


# In[16]:


X_train


# In[17]:


X_test


# #okay so far we have standardized the data now lets have a 
# 
# The neuron in the middle, applies an activation function based on the signals it receives. There are different activation functions, but one of the most common activation functions for the hidden layers is the Rectified Linear Unit (ReLU).$$\begin{align}
# f (x) = max(x,0)
# \end{align}$$
# What the ReLU does, is that it returns x, if the $\sum_{i=1}^{n} W_{i}X_{i}$ is above zero and 0 otherwise, where $x = \sum_{i=1}^{n} W_{i}X_{i}$

# In[31]:


from matplotlib import pyplot

def rectified(x):
    return max(0.0, x)
 
# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
# line plot of raw inputs to rectified outputs
pyplot.plot(series_in, series_out)
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.title('ReLU Function')
pyplot.show()


# In[34]:


import math

def sigmoid(x):
    return (1/(1 + math.exp(-x)))


# define a series of inputs
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [sigmoid(x) for x in series_in]
# line plot of raw inputs to sigmoid outputs
pyplot.plot(series_in, series_out)
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.title('Sigmoid Function')
pyplot.show()


# 
# The Neural Network below, includes an input layer with 3 neurons, a hidden layer with 4 neurons, and an output layer.
# In this example, the activation function of the hidden layer can be the ReLU function whereas for the output layer, it can be the sigmoid function.
# 
# ![image.png](attachment:image.png)

# The ann are build using keras with uses tensorflow in backend
# 
# Neural Network
# The ANN is build using the Keras framework, which uses TensorFlow in the backend.
# In the code below, the number of nodes in the input layers needs to be defined. This is the "input_dim" argument, which corresponds to the number of independent variables. Next, the "units" argument, defines the number of neurons in the hidden layer. Note that the architecture below shows a Neural Network with two hidden layers. The input layer and the first hidden layer are both defined in one line of code, while the rest of the layers are defined later.
# This means that the hidden layers have 6 neurons each with a ReLU activation function as described above while the output layer has 1 neuron and a sigmoid activation function as this is a binary classification problem.
# 
# Next, the ANN model is compiled where the optimizer, the loss, and the metric is being defined. The optimizer argument defines the optimizer that will be used in order to update the weights of the model. Amongst those are the Stochastic Gradient Descent, the Adaptive Moment Estimation (Adam), and the Root Mean Squared Propagetion (RMSProp). The most popular optimizers are the Adam and the RMSProp.
# For this problem, the binary cross entropy (logarithmic) loss is used.
# 
# $$\begin{align}
# \small
# Logarithmic Loss = - \frac{1}{n} \; \sum_{i=1}^{n} \bigg(y^{i} \; log\phi \bigg( \sum_{j=1}{m}w_{j}x_{j}^{i} \bigg) + (1-y^{i})log \bigg(1 - \phi \bigg(\sum_{j=1}^{m}w_{j}x_{j}^{i} \bigg) \bigg) \bigg)
# \end{align}$$
# where $\phi$ is the sigmoid function, i is the index of the observation and j is the index of the feature.
# 
# The metric, is the metric that will be shown, and in this case it is the accuracy.
# 
# Model Architecture

# In[35]:


import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense


# In[43]:


classifier = Sequential()
classifier.add(Dense(units = 6,activation='relu',input_dim = 11))
classifier.add(Dense(units = 6,activation='relu'))
classifier.add(Dense(units = 1,activation='sigmoid'))
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# In[45]:


print(classifier.summary())


# In[47]:


#okayy we can say that we have leaye now we can fitr the data to ann
classifier.fit(X_train,y_train,batch_size=32, epochs=50)


# In[53]:


#Furthermore, as explained previously, the model will make predictions on the test set, and a threshold will be set in order to classify the customers that are going to churn and those that are not.
#The threshold that is applied here is 0.5.

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
y_pred


# Finally, with a confusion matrix, we can see how many observations were correctly predicted. More specifically, a confusion matrix gives information about the True Positive rate, True Negative rate, False Positive rate, and False Negative rate. It is also useful in cases when the dataset is severely unbalanced. In these cases, the confusion matrix shows if the model predicts every observation into one class. The accuracy values in these unbalanced datasets can be misleading, as it can be very high.

# In[60]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print('Accuracy: ',(cm[0,0]+cm[1,1])/len(y_test)*100)


# In[61]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# by confusion metrice we can understand that 
# For this classification problem regarding customer churn, the model correctly predicted that 1528 customers will not churn and 187 customers will churn. In addition, it falsely predicted that 79 customers will churn and that 206 will not churn. The accuracy of the model is 85.75%. Below, a plot of the confusion matrix is shown.

# In[ ]:
#lets work with kfold cross validation


#wee ahve to combine keras and sklearn for kfold by using wraper

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#okayy now make an build function 
def build_classifier():
    

    classifier = Sequential()
    classifier.add(Dense(units = 6,activation='relu',input_dim = 11))
    classifier.add(Dense(units = 6,activation='relu'))
    classifier.add(Dense(units = 1,activation='sigmoid'))
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn= build_classifier,batch_size=10, epochs=100)
accuracy = cross_val_score(estimator= classifier,X = X_train,y = y_train,cv = 10,n_jobs=-1)



