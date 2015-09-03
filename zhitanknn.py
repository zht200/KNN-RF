'''Running this program alone gets the accuracy and training time 
using KNN classifier with standard dataset named ex2data1train.csv and ex2data1test.csv.'''

import time
import knn

def main(kInKnn):
	start = time.time()
	[trainingX, trainingY, testX, testY] = loadDataset('ex2data1train.csv', 'ex2data1test.csv')
	pred= knn.main(trainingX, trainingY, testX, kInKnn)
	end = time.time()
	count = 0
	for i in range(len(pred)):
		if pred[i] == testY[i]:
			count += 1
	print('accuracy: ' + str(round(count/float(len(testY))*100, 2)) + '%')
	print('training time: ' + str(round(end - start, 1)) + ' seconds')

import csv
import random
def loadDataset(filenameTrain, filenameTest):
	trainingX = []
	trainingY = []
	testX = []
	testY = []
	with open(filenameTrain, 'rt') as datafileTrain:
		trainingX = list(csv.reader(datafileTrain))
		attrLength = len(trainingX[0])
		for i in range(len(trainingX)):
			for j in range(attrLength):
				trainingX[i][j] = float(trainingX[i][j]) #converts from string to float
				if j == attrLength-1:
					trainingY.append(trainingX[i].pop(j))
	with open(filenameTest, 'rt') as datafileTest:
		testX = list(csv.reader(datafileTest))
		attrLength = len(testX[0])
		for i in range(len(testX)):
			for j in range(attrLength):
				testX[i][j] = float(testX[i][j]) #converts from string to float
				if j == attrLength-1:
					testY.append(testX[i].pop(j))
	return [trainingX, trainingY, testX, testY]

userInput = input("Please enter the number of nearest neighbors used for k nearest neighbors classifier (for example: 3):")
kInKnn = int(userInput)
main(kInKnn)