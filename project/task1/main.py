#!/usr/bin/env python3

# Helpful link: https://canvas.asu.edu/courses/73345/files/29454944?module_item_id=5342843

dataset_file_path="datasets/kdd/KDDTest+.txt"

# To load a dataset file in Python, you can use Pandas. Import pandas using the line below
import pandas as pd
# Import numpy to perform operations on the dataset
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Building FNN Training Module
from keras.models import Sequential
from keras.layers import Dense

### For testing we need an attack table that
### maps specific attack labels to their more general
### attack types
dos_types = ["apache2", "back", "land", "neptune", "mailbomb", "pod",\
                "processtable", "smurf", "teardrop", "udpstorm", "worm"]
probe_types = ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"]
u2r_types = ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", \
                "sqlattack", "xterm"]
r2l_types = ["ftp_write", "guess_passwd", "httptunnel","imap", "multihop","named",\
            "phf", "sendmail", "snmpgetattack", "spy", "snmpguess", "warezclient",\
            "warezserver", "xlock", "xsnoop", "warezmaster"]
attack_table = {}
for key in dos_types:
    attack_table[key] = "DoS"
for key in probe_types:
    attack_table[key] = "Probe"
for key in u2r_types:
    attack_table[key] = "U2R"
for key in r2l_types:
    attack_table[key] = "R2L"
attack_table["normal"] = "Normal"


'''
Takes the KDD Dataset and returns a smaller dataset containing only 
attacks in the attack_types list.
Valid attack types are: ["DoS", "Probe", "U2R", "R2L", "Normal"]
'''
def get_subset(x, y, at, select_types):
    valid_types = ["DoS", "Probe", "U2R", "R2L", "Normal"]
    for st in select_types:
        if st not in valid_types:
            print("WARN: {} is not a valid attack type! Returning entire dataset".format(at))
            return dataset
    new_x = []
    new_y = []

    for i in range(len(at)):
        if at[i] in select_types:
            new_x.append(x[i])
            new_y.append(y[i])
    new_y = np.array(new_y)
    new_x = np.array(new_x)
    return new_x, new_y

'''
Takes a dataset and returns the x and y
arrays for training/testing
'''
def prepare_data(dataset):
    x = dataset.iloc[:, 0:-2].values
    label_column = dataset.iloc[:,-2].values
    y = []
    attack_types = []
    for i in range(len(label_column)):
        attack_types.append(attack_table[label_column[i]])
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
    return x, y, attack_types

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
def init_model(inputSize):
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


### Run Tests for Task 1:
dataset = pd.read_csv(dataset_file_path, header=None)
### Attack Types: DoS, Probe, U2R, R2L
train_attacks = [["DoS","Normal"],\
                ["Probe", "Normal"],\
                ["U2R", "Normal"],\
                ["R2L","Normal"],
                ["DoS","Probe","Normal"],\
                ["DoS","U2R","Normal"],\
                ["DoS","R2L","Normal"],
                ["Probe","U2R","Normal"],\
                ["Probe","R2L","Normal"],\
                ["U2R", "R2L", "Normal"]]
test_types = ["DoS", "Probe", "U2R", "R2L"]
i = 0
results = []
for attack_types in train_attacks:
    i += 1
    print("{} attack training of {}...".format(i, len(train_attacks)))

    ### Step 1: Load Data
    dataset_file_path = "./datasets/kdd/KDDTrain+.txt"
    dataset = pd.read_csv(dataset_file_path, header=None)
    x, y, at = prepare_data(dataset)
    train_x, train_y = get_subset(x, y, at, attack_types)


    ### Step 2: Train the Model
    model = init_model(len(train_x[0]))
    train(train_x, train_y, model)

    ### Step 3: Test the model
    for test_type in test_types:
        experiment = {}

        dataset_file_path = "./datasets/kdd/KDDTrain+.txt"
        dataset = pd.read_csv(dataset_file_path, header=None)
        x, y, at = prepare_data(dataset)
        test_x, test_y = get_subset(x, y, at, [test_type])

        loss, accuracy = test(test_x, test_y, model)
        experiment["trained_on"] = attack_types
        experiment["tested_on"] = test_type
        experiment["acc"] = accuracy
        print("\t{}".format(experiment))
        results.append(experiment)

resultsFrame = pd.DataFrame(results)
resultsFrame.to_csv("./task1_results.csv")