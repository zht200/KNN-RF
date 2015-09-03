'''
This file runs the Random forest classifier on the Standard dataset. The default number of trees is 5 and the accuracy
is about 90%.
'''
import numpy as np
import pandas as pd

df = pd.read_csv('ex2data1train.csv', delimiter=',')
dataset = np.array(df).tolist()

x = []
y = []
for i in range(len(dataset)):
    y.append(float(dataset[i][len(dataset[0])-1]))
    t = []
    for j in range(len(dataset[0])-1):
        t.append(float(dataset[i][j]))
    x.append(t)


df1 = pd.read_csv('ex2data1test.csv', delimiter=',')
dataset1 = np.array(df1).tolist()

xt = []
yt = []
for i in range(len(dataset1)):
    yt.append(float(dataset1[i][len(dataset1[0])-1]))
    t = []
    for j in range(len(dataset1[0])-1):
        t.append(float(dataset1[i][j]))
    xt.append(t)

from RF import RF
rf = RF(B=5, Bagging = True)
rf.train(x, y)
yPredict = rf.predict(xt)

count = 0
for i in range(len(yPredict)):
    if yPredict[i] == yt[i]:
        count +=1
print(count/float(len(yt)))