#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all relevant packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE  
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Importing dataset and performing basic statistics on the data. 
EJdata = pd.read_csv("EireJet.csv")
pd.set_option('display.max_columns', None) 
print(EJdata.head())
print(EJdata.shape)
print(EJdata.info())
print(EJdata.describe())


# In[2]:


EJdata = EJdata.dropna(how='any',axis=0) 
print(EJdata.info())


# In[3]:


# converting 5 categorical features into numeric features and adding them back to the dataset

EJdata['Gender'] = EJdata['Gender'].map({'Female':1, 'Male':0})
EJdata['Frequent Flyer'] = EJdata['Frequent Flyer'].map({'Yes':1, 'No':0})
EJdata['Type of Travel'] = EJdata['Type of Travel'].map({'Personal Travel':1, 'Business travel':0})
EJdata['Class'] = EJdata['Class'].map({'Eco':0,'Eco Plus':1 ,'Business':2 })
EJdata['satisfaction'] = EJdata['satisfaction'].map({'neutral or dissatisfied':0, 'satisfied':1})
print(EJdata.info())


# In[4]:


# Dividing the data into label set and feature set for further analysis
X = EJdata.drop('satisfaction', axis = 1) # Feature set
Y = EJdata['satisfaction'] # Label set
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)


# In[5]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)


# In[6]:


# Dividing the processed data into training set and test set in 70,30 ratio
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)


# In[7]:


# Dividing dataset into training and test sets for application of SMOTE
X_train, X_test, Y_train, Y_test = train_test_split( X_scaled, Y, test_size = 0.3, random_state = 100)

print(X_train.shape)
print(X_test.shape)

# Now,  implementing Oversampling to balance the data; Synthetic Minority Oversampling Technique(SMOTE), printing the values
# to see the difference 
print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

smote = SMOTE(random_state = 101)
X_train,Y_train = smote.fit_sample(X_train,Y_train)

print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())


# In[8]:


# Next ,in order to tune the random forest parameter 'n_estimators' , implementing cross-validation with Grid Search
rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
grid_prm = {'n_estimators': [50, 100, 150, 200, 250, 300]}

gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_prm, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""
gd_sr.fit(X_train, Y_train)

Bestparameter = gd_sr.best_params_
print(Bestparameter)

bestresult = gd_sr.best_score_ # Mean cross-validated score of the 
print(bestresult)


# In[8]:


# Building random forest model using the tuned parameter, n estimator 150
rfc = RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='auto', random_state=1)
rfc.fit(X_train,Y_train)
featimport = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimport)

Y_pred = rfc.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[29]:


# Tuning AdaBoost parameter 'n_estimators' and using cross-validation using Grid Search method
Adaboost = AdaBoostClassifier(random_state=1)
gridparamt = {'n_estimators': [30,35,40,45,50,55,60]}

gd_sr = GridSearchCV(estimator=Adaboost, param_grid=grid_paramt, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_train, Y_train)

bestparameter = gd_sr.best_params_
print(bestparameter)

result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(result)


# In[30]:


# Building AdaBoost using the tuned parameter

Adaboost = AdaBoostClassifier(n_estimators=50, random_state=1)
Adaboost.fit(X_train,Y_train)
featimport = pd.Series(Adaboost.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimport)

Y_pred = Adaboost.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[ ]:


# Now, tuning the gradient boost parameter 'n_estimators' and using cross-validation using grid search method
Gradientboost = GradientBoostingClassifier(random_state=1)
grid_param = {'n_estimators': [100,150,200], 'max_depth' : [9,10,11,12], 'max_leaf_nodes': [8,12,16,20,24,28,32]}

grd_sr = GridSearchCV(estimator=Gradientboost, param_grid=grid_param, scoring='precision', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

grd_sr.fit(X_train, Y_train)

prime_parameters = grd_sr.best_params_
print(prime_parameters)

result = grd_sr.best_score_  #best score of the estimator , mean cross validates score
print(result)


# In[35]:


# Building Gradient Boost using the tuned parameter. 
Gradientboost = GradientBoostingClassifier(n_estimators=200, max_depth=9, max_leaf_nodes=32, random_state=1)
Gradientboost.fit(X_train,Y_train)
featimportance = pd.Series(Gradientboost.feature_importances_, index=list(X)).sort_values(ascending=False)
print(featimportance)

Y_pred = Gradientboost.predict(X_test)
print('Classification report: \n', metrics.classification_report(Y_test, Y_pred))

conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat,annot=True)
plt.title("Confusion_matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual class")
plt.show()
print('Confusion matrix: \n', conf_mat)
print('TP: ', conf_mat[1,1])
print('TN: ', conf_mat[0,0])
print('FP: ', conf_mat[0,1])
print('FN: ', conf_mat[1,0])


# In[ ]:




