# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

get_ipython().run_line_magic('matplotlib', 'inline')

#*****************************************Read the training data***********************************
print('############ Reading training data - input_train_traffic_profile.csv ########################')
df=pd.read_csv('/Users/nigurusw/Documents/Hackathon-ML/input_train_traffic_profile.csv')
df.head(3)

# converting y training data  to labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['high-BW','low-BW'])
labels = le.transform(df['requiredBW'])
print('converting y train to Labels:')
print(labels)

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(df[['hourOfDay','peakBW']],labels,
#test_size=0.2, random_state=0)

x_train = df[['hourOfDay','peakBW']]

y_train = labels

print('Printing x training data')
print(x_train)
print('Printing y training data(labels)')
print(y_train)

# x transformation
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_train)

x_train_std = sc.transform(x_train)

print('x transformed training data')
print(x_train_std)

print('Applying logistic regression for training data...')
#apply logistic regression on Training data
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=1000.0)

model.fit(x_train_std,y_train)


#**********************************************************Reading Test Data***********************************
print('#####################Reading test data - input_test_traffic_profile.csv##########################')
dft=pd.read_csv('/Users/nigurusw/Documents/Hackathon-ML/input_test_traffic_profile.csv')
dft.head(2)
x_test = dft[['hourOfDay','peakBW']]
print('x test data:')
print(x_test)


#x transform the test data
x_test_std = sc.transform(x_test)
print('transformed x test data:')
print(x_test_std)

#Y Prediction a


print('predicted Y test Data(labels):')
print(mylist)

y_test_label=[]
y_test=[]
for x_t in x_test_std:  
    myvar = np.round(model.predict_proba(x_t.reshape(1,-1)))
    #print the y predicted output
    mylist = myvar.tolist()
    print('predicted y value:')
    print(mylist)
    
    if mylist == [[0.0, 1.0]]:
        y_test.append(1)
        y_test_label.append('low-BW')
    else: 
        y_test.append(0)
        y_test_label.append('high-BW')

        
print('predicted y test vs y label for x test:')
print(y_test)
print(y_test_label)


#calculate scores
print('**************************Calculating scores *********************')
#model.score(x_test_std,y_test)

from sklearn import metrics

print('Accuracy:',metrics.accuracy_score(y_test,model.predict(x_test_std)))
print('Confusion matrix: \n', metrics.confusion_matrix(y_test,model.predict(x_test_std)))
print('classification report:\n',metrics.classification_report(y_test,model.predict(x_test_std)))

from sklearn.model_selection import cross_val_score

x=df[['hourOfDay','peakBW']]
y=df['requiredBW']
scores = cross_val_score(model,x,y,cv=5,scoring='f1_macro')
print("Crossvalidation scores:")
print(scores)
