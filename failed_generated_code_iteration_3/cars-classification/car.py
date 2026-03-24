# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 13:27:18 2020

@author: abdo
"""
# Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt

# Importing the dataset
dataset=pd.read_csv('cars.csv')
data=dataset.head()
da=dataset.info()
dataa=dataset.columns
dataset=dataset.dropna()
dataset.columns = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60',
        'year', 'brand']
dataset['cubicinches'] = pd.to_numeric(dataset['cubicinches'], errors='coerce')
dataset['weightlbs'] = pd.to_numeric(dataset['weightlbs'], errors='coerce')
dataset = dataset.dropna()

X=dataset.iloc[:,0:5].values
y=dataset.iloc[:,7].values

#visualization between brand and some important feature
a=sns.barplot(x="brand",y="mpg",data=dataset)
b=sns.barplot(x="brand",y="hp",data=dataset)
c=sns.barplot(x="brand",y="cylinders",data=dataset)
z=dataset["brand"].value_counts()

# # Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size = 0.25, random_state = 0)

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Predicting the Test set results
classifier.predict(X_test)
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)