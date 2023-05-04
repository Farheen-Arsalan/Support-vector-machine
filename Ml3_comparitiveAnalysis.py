from sklearn import datasets,metrics,svm
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = datasets.load_digits()
kernel = ['linear','poly','rbf']
ratio = ['60:40','70:30','80:20']
r = [0.5,0.3,0.2]

#divide the whole dataset into train and test 

X  = data.data # data (1797,64)
y = data.target # labels (1797)


result  = np.zeros((3,3))

for i in range(len(kernel)):
    for j in range(len(r)):
        Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=r[j],random_state=10)
        svmModel = svm.SVC(kernel = kernel[i])
        svmModel = svmModel.fit(Xtrain,ytrain) # data with its respective targets
        op =svmModel.predict(Xtest)
        
        acc = metrics.accuracy_score(ytest,op)
        result[i,j]=acc

print(result)

resultFrame = pd.DataFrame(result,index = kernel,columns=ratio)
resultFrame = resultFrame.T
resultFrame.plot(kind = 'bar')

        






















