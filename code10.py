#10. Hybrid
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Initialize and fit Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, Y_train)

# Initialize and fit Gradient Boosting Classifier
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, Y_train)

# Predict probabilities for both classifiers
rf_probs = rf.predict_proba(X_test)[:, 1]
gb_probs = gb.predict_proba(X_test)[:, 1]

# Combine predictions using a simple average
hybrid_probs = (rf_probs + gb_probs) / 2
hybrid_predicted = (hybrid_probs > 0.5).astype(int)

# Evaluate performance of the hybrid model
acc = accuracy_score(Y_test, hybrid_predicted)
f1 = f1_score(Y_test, hybrid_predicted)
precision = precision_score(Y_test, hybrid_predicted)
recall = recall_score(Y_test, hybrid_predicted)

print("Hybrid Model (Random Forest + Gradient Boosting) - Accuracy: %.2f" % acc)
print("Hybrid Model (Random Forest + Gradient Boosting) - F1 Score: %.2f" % f1)
print("Hybrid Model (Random Forest + Gradient Boosting) - Precision: %.2f" % precision)
print("Hybrid Model (Random Forest + Gradient Boosting) - Recall: %.2f" % recall)

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, hybrid_predicted)
print("Confusion Matrix:\n", conf_matrix)

# Plotting the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['H', 'P'], yticklabels=['H', 'P'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Hybrid Model')
plt.show()
