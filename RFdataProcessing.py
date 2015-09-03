'''
This is the data processing file for Random forest based on the Bank Marketing dataset.
'''
import numpy as np
import pandas as pd

df = pd.read_csv('bankTrain.csv', delimiter=';')
dataset = np.array(df).tolist()


# encode binary category 5, 7, 8, 17
for i in range(len(dataset)):
    for j in [4, 6, 7, 16]:
        if (dataset[i][j] == 'yes'):
            dataset[i][j] = '1'
        elif (dataset[i][j] == 'no'):
            dataset[i][j] = '0'
# encode categorical attribute 3 - marital
for i in range(len(dataset)):
    if (dataset[i][2] == 'married'):
        dataset[i][2] = '0'
    elif (dataset[i][2] == 'divorced'):
        dataset[i][2] = '1'
    elif (dataset[i][2] == 'single'):
        dataset[i][2] = '2'
# encode categorical attribute 4 - education
for i in range(len(dataset)):
    if (dataset[i][3] == 'unknown'):
        dataset[i][3] = '0'
    elif (dataset[i][3] == 'secondary'):
        dataset[i][3] = '1'
    elif (dataset[i][3] == 'primary'):
        dataset[i][3] = '2'
    elif (dataset[i][3] == 'tertiary'):
        dataset[i][3] = '3'
# encode categorical attribute 9 - contact
for i in range(len(dataset)):
    if (dataset[i][8] == 'unknown'):
        dataset[i][8] = '0'
    elif (dataset[i][8] == 'telephone'):
        dataset[i][8] = '1'
    elif (dataset[i][8] == 'cellular'):
        dataset[i][8] = '2'
# encode categorical attribute 16 - poutcome
for i in range(len(dataset)):
    if (dataset[i][15] == 'unknown'):
        dataset[i][15] = '0'
    elif (dataset[i][15] == 'other'):
        dataset[i][15] = '1'
    elif (dataset[i][15] == 'failure'):
        dataset[i][15] = '2'
    elif (dataset[i][15] == 'success'):
        dataset[i][15] = '3'


lists = []
x = []
y = []
for i in range(len(dataset)):
    s=""
    #delete duplicates
    for j in range(len(dataset[0])-1):
        s = s + str(dataset[i][j])
    if s not in lists:
        lists.append(s)
        y.append(float(dataset[i][len(dataset[0])-1]))
        t = []
        for j in [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
            t.append(float(dataset[i][j]))
        x.append(t)





dataset = None

df = pd.read_csv('bankTest.csv', delimiter=';')
dataset = np.array(df).tolist()

# encode binary category 5, 7, 8, 17
for i in range(len(dataset)):
    for j in [4, 6, 7, 16]:
        if (dataset[i][j] == 'yes'):
            dataset[i][j] = '1'
        elif (dataset[i][j] == 'no'):
            dataset[i][j] = '0'
# encode categorical attribute 3 - marital
for i in range(len(dataset)):
    if (dataset[i][2] == 'married'):
        dataset[i][2] = '0'
    elif (dataset[i][2] == 'divorced'):
        dataset[i][2] = '1'
    elif (dataset[i][2] == 'single'):
        dataset[i][2] = '2'
# encode categorical attribute 4 - education
for i in range(len(dataset)):
    if (dataset[i][3] == 'unknown'):
        dataset[i][3] = '0'
    elif (dataset[i][3] == 'secondary'):
        dataset[i][3] = '1'
    elif (dataset[i][3] == 'primary'):
        dataset[i][3] = '2'
    elif (dataset[i][3] == 'tertiary'):
        dataset[i][3] = '3'
# encode categorical attribute 9 - contact
for i in range(len(dataset)):
    if (dataset[i][8] == 'unknown'):
        dataset[i][8] = '0'
    elif (dataset[i][8] == 'telephone'):
        dataset[i][8] = '1'
    elif (dataset[i][8] == 'cellular'):
        dataset[i][8] = '2'
# encode categorical attribute 16 - poutcome
for i in range(len(dataset)):
    if (dataset[i][15] == 'unknown'):
        dataset[i][15] = '0'
    elif (dataset[i][15] == 'other'):
        dataset[i][15] = '1'
    elif (dataset[i][15] == 'failure'):
        dataset[i][15] = '2'
    elif (dataset[i][15] == 'success'):
        dataset[i][15] = '3'


lists = []
xt = []
yt = []
for i in range(len(dataset)):
    s=""
    #delete duplicates
    for j in range(len(dataset[0])-1):
        s = s + str(dataset[i][j])
    if s not in lists:
        lists.append(s)
        yt.append(float(dataset[i][len(dataset[0])-1]))
        t = []
        for j in [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]:
            t.append(float(dataset[i][j]))
        xt.append(t)




