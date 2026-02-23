# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 13:12:37 2024

@author: farza
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
# Read the csv file into a Pandas DataFrame
df = pd.read_csv('xgboost_on _binary.csv')
df.isnull().sum()
#EDA
df.head()
df.info()
df.describe()
df.describe(include='all')
df.shape
import seaborn as sns
import matplotlib.pyplot as plt 
pd.option_context('mode.use_inf_as_na', True)

sns.pairplot(df, kind="reg")
plt.show()
# without regression
sns.pairplot(df, kind="scatter")
plt.show()

sns.boxplot(df)

pd.crosstab(df.rank, df.admit)
# Create an instance of LabelEncoder for each categorical variable/column
le_admit = LabelEncoder()
le_rank = LabelEncoder()
# Fit and transform each categorical variable/column
df['admit'] = le_admit.fit_transform(df['admit'])
df['rank'] = le_rank.fit_transform(df['rank'])
# Print the encoded DataFrame
df['rank'].value_counts()
print(df)

# Perform one-hot encoding using get_dummies()
df_encoded = pd.get_dummies(df, columns=['admit', 'rank'], drop_first=True)
# Print the encoded DataFrame
print(df_encoded)
#EDA
df_encoded.head()
df_encoded.info()
df_encoded.describe()
df_encoded.shape
#Spliting the data 
from sklearn.model_selection import train_test_split
X = df_encoded.drop(['admit_1'], axis=1)
y = df_encoded['admit_1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
# Train the XDBoost Model
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
#make predictions
y_pred = model.predict(X_test)
# Model Evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
# 0.6 for a classification task would mean that the model correctly predicted the class label for 60%
# of the instances in the test data. 



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("accuracy", accuracy)
precision
recall
f1
roc_auc
cm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 
y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label='ROC CURVE')
plt.plot([0,1],[0,1], linestyle='--' , label = 'Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend()
plt.show()


roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc

