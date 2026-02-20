import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

#task1

file_name = 'iris.csv'

data = np.loadtxt(file_name, delimiter=',', dtype='object')

print("Data shape:",data.shape)

column_names = data[0]
data = data[1:]

features = data[:,:-1].astype(float)

labels = data[:, -1]

labels[labels == 'setosa'] = 0
labels[labels == 'versicolor'] = 1
labels[labels == 'virginica'] = 2

labels = labels.astype(float)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')

ax.set_xlabel("sepal_length")
ax.set_ylabel("sepal_width")

plt.savefig("plot1.png")
plt.show()


#task2

fig, ax = plt.subplots(4,4)

for i in range(4):
    for j in range(4):
        ax[i,j].scatter(features[:,j], features[:,i],c=labels,cmap='viridis', s=3)
        ax[i,j].set_xlabel(column_names[j])
        ax[i,j].set_ylabel(column_names[i])



plt.tight_layout()
plt.savefig("plot2.png")
plt.show()

#task3

features = data[:, 2:4].astype(float)

centroids = np.zeros((3, 2))
for i in range(3):
    centroids[i] = features[labels == i].mean(axis=0)

sample = np.array([3.1, 1.2]).reshape(1, 2)

distance =cdist(sample,centroids, metric='euclidean')

print("The distance of the found petal's features from each class centroid:", distance)

predicted_classes = np.argmin(distance)

print("Prediction:", predicted_classes)

fig, ax = plt.subplots(figsize = (7, 7))
ax.scatter(features[:,0], features[:,1], c=labels, cmap='viridis', alpha=0.15)

ax.scatter(centroids[:,0], centroids[:,1], c='red', marker='o', s=100, label='Centroids')
ax.scatter(sample[:,0], sample[:,1], c='black', marker='x', s=100, label='New petal')

ax.set_xlabel("petal_length")
ax.set_ylabel("petal_width")

plt.savefig("plot3.png")
plt.show()

# ---- Accuracy evaluation for nearest-centroid classifier ----

# Compute distances from all samples to centroids
distances = cdist(features, centroids, metric='euclidean')

# Predicted class = closest centroid
y_pred = np.argmin(distances, axis=1)

# True labels
y_true = labels.astype(int)

# Accuracy
accuracy = np.mean(y_pred == y_true)
print(f"Classification accuracy (nearest centroid): {accuracy:.4f}")