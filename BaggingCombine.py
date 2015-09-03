'''
This file is the combination of the Random forest classifier and the KNN classifier. The bagging method is used to
combine the classifiers and the final results are based on the results from these two classifier. Generally, the
combination can produce a better result than a single classifier.
'''
from RF import RF
import knn
import RFdataProcessing as RFData
import knnDataProcessing as knnData
from random import randint
import numpy as np
import time

userInput = input("Please enter the number of trees for random forest classifier (for example: 10):")
TreeNum = int(userInput)
userInput1 = input("Use bagging? (Y/N):")
if userInput1== "Y" or "y":
    isBagging = True
elif userInput1== "N" or "n":
    isBagging = False
userInput2 = input("Please enter the number of nearest neighbors used for k nearest neighbors classifier (for example: 3):")
kInKnn = int(userInput2)
userInput3 = input("Please enter the number of bagging (for example: 10):")
N = int(userInput3)

print()
print('RF: ')
print('\tnumber of trees used: ' + str(TreeNum))
print('\tuse bagging: ' + userInput1)
print('KNN: ')
print('\tnearest number: ' + str(kInKnn))
print('\tNumber of bagging: ' + str(N))
print()

def bagging(N):
    #training data
    xRF = RFData.x
    y = RFData.y
    #testing data
    xtRF = RFData.xt
    yt = RFData.yt

    xKNN = knnData.main()[0] #training X
    xtKNN = knnData.main()[2] #test X

    countYPredict = []
    for i in range(len(yt)):
        countYPredict.append(0)

    for k in range(N):  # number of bootstrapping
        x_RF = []
        y_RF = []
        x_KNN = []
        y_KNN =[]
        # bootstrapping
        for i in range(int(len(xRF) * 0.6)):
            r = randint(0, len(xRF)-1)
            tRF = []
            for j in range(len(xRF[0])-1):
                tRF.append(xRF[r][j])
            # for RF, data duplicates are not allowed
            if tRF not in x_RF:
                x_RF.append(tRF)
                y_RF.append(y[r])
            x_KNN.append(xKNN[r])
            y_KNN.append(y[r])

        # RF
        start = time.time()
        rf = RF(B= TreeNum, Bagging=isBagging)
        rf.train(x_RF,y_RF)
        pred= rf.predict(xtRF)
        end = time.time()
        count = 0
        for i in range(len(pred)):
            if pred[i] == yt[i]:
                count += 1
        print("RF, trial #"+str(k+1)+": ")
        print('\taccuracy: ' + str(round(count/float(len(yt))*100, 2)) + '%')
        print('\ttraining time: ' + str(round(end - start, 1)) + ' seconds')
        for i in range(len(pred)):
            countYPredict[i] = countYPredict[i] + pred[i]

        # KNN
        start = time.time()
        pred= knn.main(x_KNN, y_KNN, xtKNN, kInKnn)
        end = time.time()
        count = 0
        for i in range(len(pred)):
            if pred[i] == yt[i]:
                count += 1
        print("KNN, trial #"+str(k+1)+": ")
        print('\taccuracy: ' + str(round(count/float(len(yt))*100, 2)) + '%')
        print('\ttraining time: ' + str(round(end - start, 1)) + ' seconds')
        for i in range(len(pred)):
            countYPredict[i] = countYPredict[i] + pred[i]

    finalPredict = []
    for i in range(len(yt)):
        if countYPredict[i] >= N:
            finalPredict.append(1)
        else:
            finalPredict.append(0)

    count = 0
    for i in range(len(finalPredict)):
        if finalPredict[i] == yt[i]:
            count += 1
    print()
    print('After combining the classifiers by bagging: ')
    print('\taccuracy: ' + str(round(count/float(len(yt))*100, 2)) + '%')


if __name__ == '__main__':
    bagging(N)