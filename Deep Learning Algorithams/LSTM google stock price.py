#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import omly the dataset we are using for training 

dataset_train = pd.read_csv("C:/Users/trinkesh/hands-on-machine-learning-master/Part 3 - Recurrent Neural Networks/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values
 
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

#apply scaling to training the data

training_set_scale = sc.fit_transform(training_set)


#create a datastructure with 60 timesteps and 1 output

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scale[i-60:i, 0])
    y_train.append(training_set_scale[i, 0])
X_train,y_train = np.array(X_train),  np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#building the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#building LSTM layers

regressor = Sequential()
#1
regressor.add(LSTM(units =50, return_sequences=True, input_shape =  (X_train.shape[1],1)))
              
regressor.add(Dropout(0.2))

#2

regressor.add(LSTM(units =50, return_sequences=True))
              
regressor.add(Dropout(0.2))

#3

regressor.add(LSTM(units =50, return_sequences=True))
              
regressor.add(Dropout(0.2))

#4


regressor.add(LSTM(units =50, return_sequences=False))
              
regressor.add(Dropout(0.2))

#output layer

regressor.add(Dense(units = 1 ))


regressor.compile(optimizer = 'adam', loss='mean_squared_error')


#fitting the rnn to training set

regressor.fit(X_train,y_train, epochs=100, batch_size=32)


#making vizulization and prediction and vizulization


dataset_test = pd.read_csv("C:/Users/trinkesh/hands-on-machine-learning-master/Part 3 - Recurrent Neural Networks/Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values


#getting the stock

dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80 ):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

pridicted_stock_price = regressor.predict(X_test)

pridicted_stock_price = sc.inverse_transform(pridicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real google stock price')
plt.plot(pridicted_stock_price, color = 'blue', label = 'Predicted google stock price')
plt.title('google stock price prediction')
plt.xlabel('time')

plt.ylabel('google stock price')

plt.legend()
plt.show()




