#4. knn
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# Load the new dataset
data = pd.read_csv("/content/data.csv")  # Replace with the path to your dataset

# Drop non-numeric columns (e.g., IDs) if necessary
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)

# Map the target variable 'class' to numerical values
data['class'] = data['class'].map({'P': 1, 'H': 0})

# Select only numeric columns for features
X = data.drop('class', axis=1).select_dtypes(include=[np.number])
Y = data['class']

# Standardize the dataset
X_scaled = preprocessing.scale(X)
datas = pd.DataFrame(X_scaled, columns=X.columns)
datas['class'] = Y.values

# Define the target variable and features
target = "class"
X = datas.drop('class', axis=1).values
Y = datas['class'].values

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)

# Predict and evaluate the KNN model on the test set
predicted = knn.predict(X_test)
acc = accuracy_score(Y_test, predicted)
f1 = f1_score(Y_test, predicted)
precision = precision_score(Y_test, predicted)
recall = recall_score(Y_test, predicted)
#knn = KNeighborsClassifier(n_neighbors=3)

print("K-Nearest Neighbors (KNN) - Accuracy: %.2f" % acc)
print("K-Nearest Neighbors (KNN) - F1 Score: %.2f" % f1)
print("K-Nearest Neighbors (KNN) - Precision: %.2f" % precision)
print("K-Nearest Neighbors (KNN) - Recall: %.2f" % recall)

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, predicted)
print("Confusion Matrix:\n", conf_matrix)

# Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['H', 'P'], yticklabels=['H', 'P'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for K-Nearest Neighbors (KNN)')
plt.show()
