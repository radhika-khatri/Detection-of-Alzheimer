#9. Gaussian Mixture Model
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

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

# Initialize and fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2)  # Number of components/clusters
gmm.fit(X_train)

# Predict log-likelihoods for train and test sets
train_scores = gmm.score_samples(X_train)
test_scores = gmm.score_samples(X_test)

# Assign labels based on log-likelihood scores
train_predicted = (train_scores < 0).astype(int)  # Adjust threshold as necessary
test_predicted = (test_scores < 0).astype(int)  # Adjust threshold as necessary

# Evaluate performance of the GMM model
acc_train = accuracy_score(Y_train, train_predicted)
acc_test = accuracy_score(Y_test, test_predicted)
f1_train = f1_score(Y_train, train_predicted)
f1_test = f1_score(Y_test, test_predicted)
precision_train = precision_score(Y_train, train_predicted)
precision_test = precision_score(Y_test, test_predicted)
recall_train = recall_score(Y_train, train_predicted)
recall_test = recall_score(Y_test, test_predicted)

print("Gaussian Mixture Model - Train Accuracy: %.2f" % acc_train)
print("Gaussian Mixture Model - Test Accuracy: %.2f" % acc_test)
print("Gaussian Mixture Model - Train F1 Score: %.2f" % f1_train)
print("Gaussian Mixture Model - Test F1 Score: %.2f" % f1_test)
print("Gaussian Mixture Model - Train Precision: %.2f" % precision_train)
print("Gaussian Mixture Model - Test Precision: %.2f" % precision_test)
print("Gaussian Mixture Model - Train Recall: %.2f" % recall_train)
print("Gaussian Mixture Model - Test Recall: %.2f" % recall_test)

# Confusion matrix
conf_matrix_train = confusion_matrix(Y_train, train_predicted)
conf_matrix_test = confusion_matrix(Y_test, test_predicted)

print("Confusion Matrix (Train):\n", conf_matrix_train)
print("Confusion Matrix (Test):\n", conf_matrix_test)

# Plotting the confusion matrix for test set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['H', 'P'], yticklabels=['H', 'P'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Gaussian Mixture Model (Test Set)')
plt.show()

