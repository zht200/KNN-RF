'''
This file runs Random forest on the Bank Marketing dataset.
The user will need to input the number of trees and whether to use bagging in the algorithm in the console.
'''
from RF import RF
import RFdataProcessing as RFData

isBagging = True
userInput = input("Please enter the number of trees:")
TreeNum = int(userInput)
userInput1 = input("Use bagging? (Y/N):")
if userInput1== "Y":
    isBagging = True
elif userInput1== "N":
    isBagging = False

x = RFData.x
y = RFData.y
xt = RFData.xt
yt = RFData.yt

rf = RF(B=TreeNum, Bagging = isBagging)
rf.train(x, y)
yPredict = rf.predict(xt)

count = 0
for i in range(len(yPredict)):
    if yPredict[i] == yt[i]:
        count +=1
print(count/float(len(yt)))