#1. Random Forest
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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

# Initialize and evaluate KNeighborsClassifier
knn = KNeighborsClassifier()
knn_scores = cross_val_score(knn, X_train, Y_train, scoring='accuracy', cv=10).mean()
print("The mean accuracy with 10-fold cross-validation for KNN is %s" % round(knn_scores*100, 2))

# Initialize and fit RandomForestClassifier
rf = RandomForestClassifier(n_estimators=18)
rf.fit(X_train, Y_train)

# Predict and evaluate the RandomForest model on the test set
predicted_rf = rf.predict(X_test)
acc_test_rf = accuracy_score(Y_test, predicted_rf)
f1_test_rf = f1_score(Y_test, predicted_rf)
precision_test_rf = precision_score(Y_test, predicted_rf)
recall_test_rf = recall_score(Y_test, predicted_rf)

print("The accuracy on test data for RandomForest is %s" % (round(acc_test_rf, 2)))
print("The F1 score on test data for RandomForest is %s" % (round(f1_test_rf, 2)))
print("The precision on test data for RandomForest is %s" % (round(precision_test_rf, 2)))
print("The recall on test data for RandomForest is %s" % (round(recall_test_rf, 2)))

# Additional metrics and confusion matrix (optional)
conf_matrix = confusion_matrix(Y_test, predicted_rf)
print("Confusion Matrix:\n", conf_matrix)

# Plotting the confusion matrix (optional)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['H', 'P'], yticklabels=['H', 'P'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for RandomForest')
plt.show()