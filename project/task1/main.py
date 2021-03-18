#!/usr/bin/env python3

# Helpful link: https://canvas.asu.edu/courses/73345/files/29454944?module_item_id=5342843

dataset_file_path="datasets/kdd/KDDTest+.txt"

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import dataset.
dataset = pd.read_csv(dataset_file_path, header=None)
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

#Split the dataset into the training set and test set. 
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x, y, test_size =0.25, random_state = 0)

# Perform feature scaling using StandardScaler method
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Scaling to the range [0,1]
x_train = sc.fit_transform(xtrain)
x_test =  sc.fit_transform(xtest)

# Building FNN Training Module
from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
classifier = Sequential()

# Build neural network using 6 nodes as input layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(x_train[0])))

# Add second hidden layer
classifier.add(Dense(units=6, kernel_initializer = 'uniform', activation = 'relu'))

# Add output layer, 1 node
classifier.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# FNN is constructed, time to compile NN. Use gradient decent algorithm 'adam'
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

# Run training with batch size equal to 10 and 10 epochs will be performed
classiferHistory = classifier.fit(xtrain,ytrain,batch_size=10,epochs=10)

# Evaluate training results, showing loss and accuracy of training on the given dataset
loss, accuracy = classifier.evaluate(xtrain,ytrain)
print("Print the loss and accuracy of the model on the dataset")
print("Loss [0,1]: %.4f" % (loss), "Accuracy [0,1]: %.4f" % (accuracy))