import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import tensorflow as tf 
from tensorflow.keras.models import Sequential

#read data
df = pd.read_csv(r"C:\Users\giga_\Desktop\Radmila_Gigovic_660_2017_VI\heart\heart_dataset_.csv")

print(df)
print("\n")

df["target"].unique()
print("\n Possible values of attribute target : ")
print(df["target"].unique())

df.head()


#1.figura-AGE
df.groupby('target').mean()
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.show()

#2.FIGURA-SEX
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#3.FIGURA-CHEST PAIN 
pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#4.FIGURA-REST BLOOD PRESSURE
pd.crosstab(df.trestbps,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To Rest Blood Pressure')
plt.xlabel('Rest Blood Pressure')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#5.FIGURA- Serum Cholestoral   nepregledno
pd.crosstab(df.chol,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Serum Cholestoral')
plt.xlabel('Serum Cholestoral')
#plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#6.FIGURA- Fasting Blood Sugar
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl)(1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#7.FIGURA- ResElectrocardiographic	
pd.crosstab(df.restecg,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To ResElectrocardiographic	')
plt.xlabel('ResElectrocardiographic')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#8.FIGURA- Max Heart Rate  nepregledno	
pd.crosstab(df.thalach,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To Max Heart Rate')
plt.xlabel('Max Heart Rate')
#plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#9.FIGURA- Exercise Induced	
pd.crosstab(df.exang,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To Exercise Induced')
plt.xlabel('ExerciseInduced')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#10.FIGURA- Oldpeak	
pd.crosstab(df.oldpeak,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency According To Oldpeak')
plt.xlabel('Oldpeak')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#11.FIGURA- Slope	
pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('Slope')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#12.FIGURA- MajorVessels	
pd.crosstab(df.ca,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency for Major Vessels')
plt.xlabel('MajorVessels')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#13.FIGURA- Thal	
pd.crosstab(df.thal,df.target).plot(kind="bar",figsize=(10,5),color=['#11A5AA','#AA1190'])
plt.title('Heart Disease Frequency for Thal')
plt.xlabel('Thal')
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()


#14.FIGURA- TARGET
df.target.value_counts()
plt.title('Heart Disease Absence or Presence')
sns.countplot(x="target", data=df, palette="bwr")
plt.xticks(rotation = 0)
plt.legend(["ABSENCE of heart disease", "PRESENCE of heart disease"])
plt.ylabel('Count')
plt.show()

#Percentage of people withouth and with heart disease
countNoDisease = len(df[df.target == 1])
countHaveDisease = len(df[df.target == 2])
print("Percentage of Patients Without Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients With Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


#Percentage of female and male patients
countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))




#priprema podataka za obucavanje

y = df.target.values
x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)


x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
print('\n')

accuracies = {}

lr = LogisticRegression(solver="lbfgs")
lr.fit(x_train.T,y_train.T)
print("Logistic regression-accuracy: {}%".format(round(lr.score(x_test.T,y_test.T)*100,2)))
print('\n')

# KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

#print("{} Algoritam K najblizih suseda: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
# try ro find best k value
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(x_train.T, y_train.T)
    scoreList.append(knn2.score(x_test.T, y_test.T))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()

acc = max(scoreList)*100
accuracies['KNN'] = acc
print("K-nearest neighbors-accuracy score: {:.2f}%".format(acc))
print('\n')



#SVC
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

acc = svm.score(x_test.T,y_test.T)*100
accuracies['SVM'] = acc
print("SVM algorithm-accuracy score: {:.2f}%".format(acc))
print('\n')


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Naive Bayes-accuracy score: {:.2f}%".format(acc))
print('\n')



#Decision Tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree-accuracy score: {:.2f}%".format(acc))
print('\n')

#RANDOM FOREST
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T,y_test.T)*100
accuracies['Random Forest'] = acc
print("Random Forest-accuracy score: {:.2f}%".format(acc))



#POREDJENJE MODELA
colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


#konfuzione matrice

# Predicted values
y_head_lr = lr.predict(x_test.T)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train.T, y_train.T)
y_head_knn = knn3.predict(x_test.T)
y_head_svm = svm.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)
y_head_rf = rf.predict(x_test.T)


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)



plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()





