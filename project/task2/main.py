#!/usr/bin/env python3

'''
CSE 548 Project 2 Task 2

Builds neural network models and runs tests according
to subtask 2 in project 2 description

usage:
from project directory:
python ./task2/main.py

Authors:
Kirtus Leyba, Austin Derbique, Ryan Schmidt

'''

### imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.preprocessing.sequence import TimeseriesGenerator
import sys

'''
For the KDD NDS dataset
Takes a dataset and returns the x and y
arrays for training/testing
'''
def prepare_data_kdd(dataset):
    x = dataset.iloc[:, 0:-2].values
    label_column = dataset.iloc[:,-2].values
    y = []
    for i in range(len(label_column)):
        if label_column[i] == 'normal':
            y.append(0)
        else:
            y.append(1)
    # Convert i-st to array
    y = np.array(y)

    ct = ColumnTransformer(
        # The column numbers  to  be  transformed  ([1 ,  2 ,  3]  representsthree  columns  to  be  transferred )
        [('onehotencoder', OneHotEncoder(), [1,2,3])],
        # Leave  the  r e s t  of  the  columns  untouched
        remainder='passthrough'
    )
    x = np.array(ct.fit_transform(x), dtype=float)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y

def train_test_split(x,y,fraction):
	r = int(fraction*len(x))
	train_x = x[:r]
	test_x = x[r:]
	train_y = y[:r]
	test_y = y[r:]
	return test_x, train_x, test_y, train_y

def get_ts_gen(x, y, lookback):
	return TimeseriesGenerator(x,y, length=lookback)

'''
Train a model, returns fitting history
'''
def train(x, y, model):
    classiferHistory = model.fit(x,y,batch_size=10,epochs=10)
    return classiferHistory

'''
Test a model, returns loss and accuracy
'''
def test(x,y, model):
    loss, accuracy = model.evaluate(x,y)
    print("Print the loss and accuracy of the model on the dataset")
    print("Loss [0,1]: %.4f" % (loss), "Accuracy [0,1]: %.4f" % (accuracy))
    return loss, accuracy

'''
Initialized the Neural Net Model
returns the model
'''
def init_ann_model(inputSize):
    # Initialize the ANN
    classifier = Sequential()

    # Build neural network using 6 nodes as input layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputSize))

    # Add second hidden layer
    classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))

    # Add output layer, 1 node
    classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    # FNN is constructed, time to compile NN. Use gradient decent algorithm 'adam'
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    return classifier

def init_rnn_model(inputSize, rnnType, lookback):
	if(rnnType not in ["simple", "gru", "lstm"]):
		print("ERROR: Invalid RNN type provided")
		sys.exit(0)

	if rnnType == "simple":
		classifier = Sequential()

		### Add a simpleRNN layer
		classifier.add(SimpleRNN(units = 16, \
			kernel_initializer = 'uniform', \
			input_shape=(lookback, inputSize)))

		### Add output layer, 1 node
		classifier.add(Dense(units=1, \
			kernel_initializer = 'uniform', \
			activation = 'sigmoid'))

		classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

		return classifier

### Load Dataset and Test RNN Model
dataset_file_path="datasets/kdd/KDDTest+.txt"
dataset = pd.read_csv(dataset_file_path, header=None)
x, y = prepare_data_kdd(dataset)
train_x, test_x, train_y, test_y = train_test_split(x,y,0.75)

lookback = 10

train_gen = get_ts_gen(train_x, train_y, lookback)
test_gen = get_ts_gen(test_x, test_y, lookback)

#train
model = init_rnn_model(len(train_x[0]), "simple", lookback)
model.fit(train_gen, epochs=10, verbose=1)

#test
loss, accuracy = model.evaluate(test_gen)
print("Print the loss and accuracy of the model on the dataset")
print("Loss [0,1]: %.4f" % (loss), "Accuracy [0,1]: %.4f" % (accuracy))