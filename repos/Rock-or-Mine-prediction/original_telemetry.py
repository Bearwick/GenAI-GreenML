
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loding the dataset to pandas dataframe
sonar_data= pd.read_csv('Sonar Data.csv',header=None)

#.head() prints first 5 rows of data sets
#sonar_data.head()

# Returns the no of rows and columns of data sets 
# sonar_data.shape

sonar_data.describe()  #describe  --> Stastecial measure of data sets

sonar_data[60].value_counts()

#  M --> Mine
#  R --> Rock


# Grouping data by rock and mine and finding mean of both rock and mine column
sonar_data.groupby(60).mean()

# Seprating data and lables 

X = sonar_data.drop(60, axis=1)
Y=sonar_data[60]

# Spliting data into traning and test data

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.7,stratify=Y,random_state=1)
print(X.shape,X_test.shape,X_train.shape)

model=LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction=model.predict(X_train)
traning_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of traning data is : ",traning_data_accuracy)

X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of test data is : ",test_data_accuracy)

print(f"ACCURACY={test_data_accuracy:.6f}")


