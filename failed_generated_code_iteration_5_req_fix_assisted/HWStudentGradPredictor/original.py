import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/students_graduate_predict.csv', delimiter=';')
df

df['Graduated (target)'].hist()

X = df.drop(columns=['Graduated (target)'])
X

Y = df['Graduated (target)']
Y

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15,random_state=1)
x_test
# x_train

model = MLPClassifier(hidden_layer_sizes=[5, 7], max_iter=800)
model.fit(x_train, y_train)

#model.predict(X.iloc[0].to_frame().T)
model.predict(X.iloc[[0, 1, 2, 3]])

y_pred_lr = model.predict(x_test)
y_test

pd.DataFrame({'Предсказанные': y_pred_lr, 'Истинные': y_test})

accuracy_score(y_test, y_pred_lr)

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, cmap='Blues', ax=ax)
ax.set_xlabel('Предсказанные')
ax.set_ylabel('Истинные')
ax.set_title('Матрица ошибок')
plt.show()

pred = [precision_score(y_test, y_pred_lr),
recall_score(y_test, y_pred_lr),
f1_score(y_test, y_pred_lr)]
pred