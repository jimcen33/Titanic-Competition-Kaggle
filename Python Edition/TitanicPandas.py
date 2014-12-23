# -*- coding: utf-8 -*-
"""
Titanic Disaster Prediction from Kaggle 

Created on Fri Dec 19 08:39:20 2014

@Organized by JimCen
Original source from Kaggle
"""

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)
df.info()
df.describe()
import pylab as P
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()
df['Gender'] = 4
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Filled in the missing Age values with their class's median age
median_ages = np.zeros((2,3))
median_ages
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
 
median_ages

df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

#Create a feature that records whether the Age was originally missing
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass


#Drop the objects which are not used in the ML implementation 
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked',"AgeIsNull"], axis=1) 
df = df.drop(['Age'], axis=1)
df = df.drop(["PassengerId"],axis=1)
df = df.dropna()

#The final step is to convert it into a Numpy array. Pandas can always
# send back an array using the .values method. Assign to a new variable, 
#train_data:
train_data = df.values

################Fill up and Clean data for Test dataset##############

# For .read_csv, always use header=0 when you know row 0 is the header row
dfTest = pd.read_csv('test.csv', header=0)
dfTEST = pd.read_csv('test.csv', header=0)
dfTest.info()
dfTest.describe()

dfTest['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
dfTest['Gender'] = 4
dfTest['Gender'] = dfTest['Sex'].map( lambda x: x[0].upper() )
dfTest['Gender'] = dfTest['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#Filled in the missing Age values with their class's median age
median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = dfTest[(dfTest['Gender'] == i) & \
                              (dfTest['Pclass'] == j+1)]['Age'].dropna().median()
 

dfTest['AgeFill'] = dfTest['Age']

for i in range(0, 2):
    for j in range(0, 3):
        dfTest.loc[ (dfTest.Age.isnull()) & (dfTest.Gender == i) & (dfTest.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

#Create a feature that records whether the Age was originally missing
dfTest['AgeIsNull'] = pd.isnull(dfTest.Age).astype(int)

dfTest['FamilySize'] = dfTest['SibSp'] + dfTest['Parch']
dfTest['Age*Class'] = dfTest.AgeFill * dfTest.Pclass

#Drop the objects which are not used in the ML implementation 
dfTest = dfTest.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked',"AgeIsNull"], axis=1) 
dfTest = dfTest.drop(['Age'], axis=1)
dfTest = dfTest.drop(["PassengerId"],axis=1)
dfTest = dfTest.dropna()

#The final step is to convert it into a Numpy array. Pandas can always
# send back an array using the .values method. Assign to a new variable, 
#test_data:
test_data = dfTest.values



#################Implement Random forest ML algorithm##############

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)

