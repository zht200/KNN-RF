'''
Processes the bank data for KNN classifier
'''
# to install modules on Mac OS X: python3.4 -m pip install PackageName
import numpy as np
import pandas as pd

def main():
	df = pd.read_csv('bankTrain.csv', delimiter = ';')
	trainDataset = np.array(df).tolist()
	[x, y] = process(trainDataset) #training x and y

	df = pd.read_csv('bankTest.csv', delimiter = ';')
	testDataset = np.array(df).tolist()
	[xt, yt] = process(testDataset) #test x and y

	return [x, y, xt, yt]

def process(dataset):
	# encode binary category 5, 7, 8, 17
	for i in range(len(dataset)):
		for j in [4, 6, 7, 16]:
			if (dataset[i][j] == 'yes'):
				dataset[i][j] = 1
			elif(dataset[i][j] == 'no'):
				dataset[i][j] = 0
	# encode categorical attribute 3 - marital
	for i in range(len(dataset)):
		if (dataset[i][2] == 'married'):
			dataset[i] += [1, 0, 0]
		elif (dataset[i][2] == 'divorced'):
			dataset[i] += [0, 1, 0]
		elif (dataset[i][2] == 'single'):
			dataset[i] += [0, 0, 1]
	# encode categorical attribute 4 - education
	for i in range(len(dataset)):
		if (dataset[i][3] == 'unknown'):
			dataset[i] += [1, 0, 0, 0]
		elif (dataset[i][3] == 'secondary'):
			dataset[i] += [0, 1, 0, 0]
		elif (dataset[i][3] == 'primary'):
			dataset[i] += [0, 0, 1, 0]
		elif (dataset[i][3] == 'tertiary'):
			dataset[i] += [0, 0, 0, 1]
	# encode categorical attribute 9 - contact
	for i in range(len(dataset)):
		if (dataset[i][8] == 'unknown'):
			dataset[i] += [1, 0, 0]
		elif (dataset[i][8] == 'telephone'):
			dataset[i] += [0, 1, 0]
		elif (dataset[i][8] == 'cellular'):
			dataset[i] += [0, 0, 1]
	# encode categorical attribute 16 - poutcome
	for i in range(len(dataset)):
		if (dataset[i][15] == 'unknown'):
			dataset[i] += [1, 0, 0, 0]
		elif (dataset[i][15] == 'other'):
			dataset[i] += [0, 1, 0, 0]
		elif (dataset[i][15] == 'failure'):
			dataset[i] += [0, 0, 1, 0]
		elif (dataset[i][15] == 'success'):
			dataset[i] += [0, 0, 0, 1]

	outcome = []
	# put the output variable out
	for i in range(len(dataset)):
		outcome.append(dataset[i].pop(16))
		for j in range(len(dataset[i])-1, -1, -1):
			if type(dataset[i][j]) is str:
				dataset[i].pop(j)

	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			dataset[i][j] = float(dataset[i][j])

	from sklearn import preprocessing
	# to install on Mac OS X: python3.4 -m pip install scikit-learn
	dataset_scaled = preprocessing.scale(dataset).tolist()

	return [dataset_scaled, outcome]


