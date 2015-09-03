'''
The KNN classifier
'''
def main(trainingX, trainingY, testX, k):
	resultSet = []
	for i in range(len(testX)):
		nearestK = getNearestK(testX[i], trainingX, trainingY, k)
		resultSet.append(getResult(nearestK))
	return resultSet

import operator 
def getNearestK(singleTestX, trainingX, trainingY, k):
	distance = [] #has the form [[X, distance], [X, distance],...]
	for i in range(len(trainingX)):
		distance.append([trainingY[i], euclideanDistance(singleTestX, trainingX[i])])
	distance.sort(key=operator.itemgetter(1))
	nearestK = []
	for i in range(k):
		nearestK.append(distance[i][0])
	return nearestK

import math
def euclideanDistance(X1, X2):
	distance = 0
	for i in range(len(X1)):
		distance += pow((X1[i] - X2[i]), 2)
	return math.sqrt(distance)

def getResult(nearestK):
	result = {}
	for i in range(len(nearestK)):
		answer = nearestK[i]
		if answer in result:
			result[answer] += 1
		else:
			result[answer] = 1
	sortedResult = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
	return sortedResult[0][0]

