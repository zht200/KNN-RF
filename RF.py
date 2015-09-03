'''
This is the implementation of Random forest classifier.
'''
from random import randint
from numpy.random import shuffle
import numpy as np

class RF(object):

    def __init__(self, B, Bagging):
        self.data = None
        # a single tree
        self.tree = None
        # ID of trees
        self.count = 0
        # total number of trees
        self.B = B
        # whether use bagging
        self.Bagging = Bagging
        # the ensemble of trees
        self.forest = None

    def train(self, x, y):
        # combine x,y into one dataset
        data = np.zeros([len(x), len(x[0])+1])
        data[:, 0:len(x[0])] = x
        data[:, len(x[0])] = y
        self.forest =[]
        B = self.B
        for b in range(0, B):
            if self.Bagging:
                self.count = 0
                shuffle(data)
                # bootstrapping: select 60% data points randomly
                randomdata = data[:(len(x) * 0.6)]
                # node of tree, [0]: split value, [1] split attribute, [2] left child ID, [3] right child ID
                self.tree = np.zeros((randomdata.size, 4))
                self.buildtree(randomdata)
                # add the new tree to the forest
                self.forest.append(self.tree[0:self.count+1])
                self.tree = None
                randomdata = None
            else:
                self.count = 0
                self.tree = np.zeros((data.size, 4))
                self.buildtree(data)
                self.forest.append(self.tree[0:self.count+1])
                self.tree = None

    def buildtree(self, data):
        row = data.shape[0]
        col = data.shape[1]
        # record parent node ID
        parent = self.count

        # if not the min node
        if row > 1:
            # choose one attribute randomly
            attribute = randint(0, col - 2)

            # choose one value as the split criteria
            val = 0
            for k in range(5):
                val = val + data[randint(0, row-1), attribute]
            splitValue = val/5
            # record the info into the current node, -1 means no child
            self.tree[self.count, :] = [splitValue, attribute, self.count+1, -1]

            # split the current node:
            left = np.zeros((row,col))
            right = np.zeros((row,col))
            l = 0
            r = 0
            lcount = 0
            rcount = 0
            for i in range(0, row):
                if data[i, attribute] <= splitValue:
                    left[l,:] = data[i,:]
                    l += 1
                    lcount += 1
                else:
                    right[r,:] = data[i,:]
                    r += 1
                    rcount += 1
            # build left tree
            self.count += 1
            self.buildtree(left[0:lcount])
            # build right tree
            if (rcount > 0):
                self.count += 1
                rightTreeId = self.count
                self.buildtree(right[0:rcount])
                self.tree[parent, 3] = rightTreeId
        # reach the min node
        elif row == 1:
            self.tree[self.count,:] = [data[0, col-1], -1, -1, -1]


    def predict(self, test):
        B = self.B
        y = []
        for i in range(len(test)):
            count1 = 0
            count0 = 0
            for b in range(B):
                tree = self.forest[b]
                j = 0
                # find the terminal node
                while tree[j, 1] != -1:
                    if test[i][int(tree[j,1])] <= tree[j, 0]:
                        j = tree[j, 2] # go left
                    else:
                        j = tree[j, 3] # go right
                if tree[j, 0] == 0:
                    count0 += 1
                if tree[j, 0] == 1:
                    count1 += 1
            if count0 > count1:
                y.append(0)
            else:
                y.append(1)
        return y