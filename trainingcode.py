# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# List files in the current directory to verify dataset presence
print("Files in current directory:")
for filename in os.listdir('.'):
    print(filename)

try:
    # Load the dataset from the local CSV file
    df = pd.read_csv("dataset_phishing.csv")

    print("Dataset loaded successfully!")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    print("\nShape of the dataset (rows, columns):")
    print(df.shape)

    print("\nInformation about the dataset (data types, non-null counts):")
    df.info()

    print("\nDistribution of the target variable (status):")
    print(df['status'].value_counts())
except FileNotFoundError:
    print("Error: 'dataset_phishing.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script, or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("\n--- Data Cleaning and Preprocessing ---")

# Drop the 'url' column as it's not needed for the model
if 'url' in df.columns:
    df = df.drop("url", axis=1)
    print("'url' column dropped.")
else:
    print("'url' column not found.")

# Encode the 'status' column using LabelEncoder
if 'status' in df.columns:
    label_encoder = LabelEncoder()
    df["status_encoding"] = label_encoder.fit_transform(df["status"])
    status_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Status column encoded. Mapping: {status_mapping}")
    # Drop the original 'status' column
    df = df.drop("status", axis=1)
    print("Original 'status' column dropped.")
else:
    print("'status' column not found or already processed.")


# Handle -1 values in 'domain_age' and 'domain_registration_length'
print("\nHandling -1 values in 'domain_age' and 'domain_registration_length'...")

# Calculate the mean of the relevant columns excluding -1
mean_domain_age = df[df['domain_age'] != -1]['domain_age'].mean()
mean_domain_registration_length = df[df['domain_registration_length'] != -1]['domain_registration_length'].mean()

# Replace -1 values with the calculated means
df['domain_age'] = df['domain_age'].replace(-1, mean_domain_age)
df['domain_registration_length'] = df['domain_registration_length'].replace(-1, mean_domain_registration_length)

print(f"Replaced -1 in 'domain_age' with mean: {mean_domain_age:.2f}")
print(f"Replaced -1 in 'domain_registration_length' with mean: {mean_domain_registration_length:.2f}")


# Define features (X) and target (y)
X = df.drop('status_encoding', axis=1)
y = df['status_encoding']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set features (X_train) shape: {X_train.shape}")
print(f"Testing set features (X_test) shape: {X_test.shape}")
print(f"Training set target (y_train) shape: {y_train.shape}")
print(f"Testing set target (y_test) shape: {y_test.shape}")

print("\nPreprocessing complete. Data is ready for model training.")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

print("\n--- Keras Neural Network Model Training ---")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# Define the Keras Sequential model
model_keras = Sequential()

# Add hidden layers with ReLU activation
model_keras.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model_keras.add(Dense(units=32, activation='relu')) # Another hidden layer with ReLU

# Add another hidden layer with Sigmoid activation (optional, can mix and match)
model_keras.add(Dense(units=16, activation='sigmoid'))

# Add the output layer with Sigmoid activation for binary classification
model_keras.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model_keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Keras model compiled.")

# Train the model
print("Training Keras Neural Network...")
history = model_keras.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
print("Model training complete.")

# Evaluate the model
print("Evaluating Keras Model...")
y_pred_keras_prob = model_keras.predict(X_test_scaled)
y_pred_keras = (y_pred_keras_prob > 0.5).astype("int32")

accuracy_keras = accuracy_score(y_test, y_pred_keras)
precision_keras = precision_score(y_test, y_pred_keras)
recall_keras = recall_score(y_test, y_pred_keras)
f1_keras = f1_score(y_test, y_pred_keras)
conf_matrix_keras = confusion_matrix(y_test, y_pred_keras)

print(f"\nAccuracy (Keras): {accuracy_keras:.4f}")
print(f"Precision (Keras): {precision_keras:.4f}")
print(f"Recall (Keras): {recall_keras:.4f}")
print(f"F1-Score (Keras): {f1_keras:.4f}")

print("\nConfusion Matrix (Keras):")
print(conf_matrix_keras)

print("\n--- Confusion Matrix Breakdown (Keras) ---")
TN_keras, FP_keras, FN_keras, TP_keras = conf_matrix_keras.ravel()
print(f"True Negatives (Legitimate Correctly Classified): {TN_keras}")
print(f"False Positives (Legitimate Classified as Phishing): {FP_keras}")
print(f"False Negatives (Phishing Classified as Legitimate): {FN_keras}")
print(f"True Positives (Phishing Correctly Classified): {TP_keras}")