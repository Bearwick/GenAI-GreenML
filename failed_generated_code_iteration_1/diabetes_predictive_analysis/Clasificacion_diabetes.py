import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("diabetes.csv")
print(data.head())
print(data.describe().T)
      

data_copy = data.copy()
print(data_copy.columns)

data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
print(data_copy.info())

data_copy.hist(figsize=(20, 15))
plt.show()

cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    plt.boxplot(data_copy[col].dropna(), vert = False)
    plt.title(col)
    plt.show()
    
data_copy['Glucose'] = data_copy['Glucose'].fillna(data_copy['Glucose'].mean())
data_copy['BloodPressure'] = data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].median())
data_copy['SkinThickness'] = data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median())
data_copy['Insulin'] = data_copy['Insulin'].fillna(data_copy['Insulin'].median())
data_copy['BMI'] = data_copy['BMI'].fillna(data_copy['BMI'].median())
print(data_copy.info())
print(data_copy.describe())

plt.figure(figsize=(20, 15))
sns.heatmap(data_copy.corr(), annot = True)
plt.show()

x = data_copy.drop('Outcome', axis=1).values
y= data_copy['Outcome'].values

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)
y_pred = knn.predict(x_test_scaled)
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
print('F1 Score: ', f1_score(y_test, y_pred))

test_f1 = []
test_pre = []
test_rec = []
test_score = []

for i in range(1, 16):
    knn = KNeighborsClassifier(i)
    knn.fit(x_train_scaled, y_train)
    y_pred = knn.predict(x_test_scaled)
    
    test_f1.append(round(f1_score(y_test, y_pred, average='weighted') *100, 2))
    test_pre.append(round(precision_score(y_test, y_pred, average='weighted') *100, 2))
    test_rec.append(round(recall_score(y_test, y_pred, average='weighted') *100, 2))
    test_score.append(round(knn.score(x_test, y_test) *100, 2))
    
plt.figure(figsize=(12, 8))
plt.plot(range(1, 16), test_f1, label = 'F1 Score', marker='o')
plt.plot(range(1, 16), test_pre, label = 'Precision', marker='o')
plt.plot(range(1, 16), test_rec, label = 'Recall', marker='o')
plt.plot(range(1, 16), test_score, label = 'Score', marker='o')
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score

for i in range(1, 16):
    knn = KNeighborsClassifier(i)
    scores = cross_val_score(knn, x_train, y_train, cv = 5)
    print('KNN: ', i)
    print('Scores:', scores)
    print('Accuracy: %0.2f (+/- %0.2f)' %(scores.mean(), scores.std() * 2))
    
from sklearn.model_selection import RandomizedSearchCV    
param_dist = {'n_neighbors': np.arange(1, 16), 
              'weights': ['uniform', 'distance']}

knn = KNeighborsClassifier()
random_search = RandomizedSearchCV(knn, param_dist, n_iter = 10, cv = 5)
random_search.fit(x_train, y_train)

print('Best parameters: ', random_search.best_params_)
print('Best score: ', random_search.best_score_)


                          