# Artifical Neural Networks (ANN)

# Importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

ds = pd.read_csv('Churn_MOdelling.csv')
X = ds.iloc[:,3:-1].values
y = ds.iloc[:,-1].values

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X[:,1] = le_x.fit_transform(X[:,1])
X[:,2] = le_x.fit_transform(X[:,2])
ohe_x = OneHotEncoder(categorical_features = [1])
X = ohe_x.fit_transform(X).toarray()

# Avoiding the dummy variable

X = X[:,1:]

# Splitting the data into the training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# This is compulsory as there will be a high volume of computations. Highly computationally intensive computations 
# and parallel computions require Feature Scaling to ease all the calculations. We also do not want one independent 
# variable dominating another independent variable.

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Importing the keras libraries and packages
# Theano - Numerical computations library that runs on CPU and GPU
# Tensorflow - Numerical computations library that runs on CPU and GPU
# Keras - Wraps Theano and Tensorflow libraries together. The Keras library is used to build deep learning models like 
# deep neural networks efficiently in only a few lines of code. 

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN, as it is a sequence of layers
# We are making a classification ANN. The classifier below is the ANN we will build and it is an object of the sequential class.

classifier = Sequential()

# Adding the input layer and the first hidden layer
# output_dim are the number of nodes in the hidden layer which is usually 1/2*(Nodes in the input layer + Nodes in the output layer)
# Initialise 'init' uses the uniform distribution function to initialise the weights close to zero.
# 'relu' is the Rectifier Activation function in the hidden layer
# input_dim is the number of independent variables (input nodes)

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
# output_dim is the number of nodes in the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# output_dim is 1 as we have a binary output (0 for no or 1 for yes)
# The activation function is now sigmoid as we want the probability of the occurence of yes (or 1)
# If we have three or more categories for our output variable our output_dim >= 3 and our activation function = softmax

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN - Applying the stochastic gradient descent on the whole ANN
# optimizer = 'adam': is the algorithm used to find the optimal set of weights in the Neural Networks
# loss = 'binary_crossentropy': Binary outcomes (2 categories)
# loss = 'categorical_crossentropy': Three or more categories
# metrics = ['accuracy']: The criterion we use to evaluate our model

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
# Batch size - The number of observations per epoch
# epochs - The number of epochs (number of times the whole of the dataset is passed through the ANN)

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
# y_pred is the probability of each individual leaving the bank 0< y_pred <1
# y_pred > 0.5 will convert the probabilities into True if > 0.5 and False if <=0.5

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)