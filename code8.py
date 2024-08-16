#8. Autoencoder
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Model
from keras.layers import Input, Dense

# Load and preprocess your dataset
data = pd.read_csv("/content/data.csv")  # Replace with your dataset path

# Drop non-numeric columns (e.g., IDs) if necessary
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)

# Map the target variable 'class' to numerical values
data['class'] = data['class'].map({'P': 1, 'H': 0})

# Select only numeric columns for features
X = data.drop('class', axis=1).select_dtypes(include=[np.number])
Y = data['class']

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Autoencoder architecture
input_dim = X_train.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)  # Example architecture, adjust as needed
encoded = Dense(32, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Extract features using the encoder part of the autoencoder
encoder = Model(inputs=input_layer, outputs=encoded)
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

# Now use these encoded features for classification
# Example: Logistic Regression as classifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(encoded_features_train, Y_train)

# Predictions and evaluation
predicted = lr.predict(encoded_features_test)
acc = accuracy_score(Y_test, predicted)
f1 = f1_score(Y_test, predicted)
precision = precision_score(Y_test, predicted)
recall = recall_score(Y_test, predicted)

print("Autoencoder + Logistic Regression - Accuracy: %.2f" % acc)
print("Autoencoder + Logistic Regression - F1 Score: %.2f" % f1)
print("Autoencoder + Logistic Regression - Precision: %.2f" % precision)
print("Autoencoder + Logistic Regression - Recall: %.2f" % recall)

# Confusion matrix
conf_matrix = confusion_matrix(Y_test, predicted)
print("Confusion Matrix:\n", conf_matrix)
