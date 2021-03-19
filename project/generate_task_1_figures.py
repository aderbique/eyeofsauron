import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt

df = pd.read_csv("task1_results.csv")
df["trained_on"] = df["trained_on"].apply(eval)

i = 0
line = ""
for index, row in df.iterrows():
    i += 1
    line += " {:.3f} &".format(int(row["acc"] * 1000)/1000.0)
    if i == 4:
        line = line[0:-2]
        line += " \\\\ \\hline"
        start = ""
        l = list(row["trained_on"])
        for t in l:
            start += "{}, ".format(t)
        start = start[0:-2]
        line = start + " & " + line
        print(line)
        line = ""
        i = 0

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

dataset_file_path = "./datasets/kdd/KDDTrain+.txt"
dataset = pd.read_csv(dataset_file_path, header=None)
x, y, at = prepare_data(dataset)

pca = decomposition.PCA(n_components=2)
pca.fit(x)
result = pca.transform(x)
color_dict = {"DoS":"red", "Probe":"blue", "U2R":"green", "R2L":"yellow"}

plt.figure()
for attack in color_dict.keys():
    xp = []
    yp = []
    for i in range(len(x)):
        if at[i] == attack:
            xp.append(result[i][0])
            yp.append(result[i][1])
    plt.scatter(xp,yp,c=color_dict[attack], alpha=0.2, label=attack)
plt.xticks([])
plt.yticks([])
plt.legend()
plt.savefig("pca.png")