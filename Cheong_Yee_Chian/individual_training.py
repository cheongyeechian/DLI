import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV # Changed from GridSearchCV
import numpy as np
from scipy.stats import uniform # Needed for RandomizedSearchCV's distribution
from scipy.stats import loguniform # Needed for RandomizedSearchCV's distribution

# Ensure the dataset is downloaded if not already present
if not os.path.exists("dataset_phishing.csv"):
    print("Dataset not found. Please ensure the file exists in the current directory.")
    exit()

# Load the dataset
try:
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
    print("Error: 'dataset_phishing.csv' not found after download attempt.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()


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



print("\n--- Logistic Regression Model Training with Enhancements ---")

# Scale the features
scaler_lr = StandardScaler()
X_train_scaled_lr = scaler_lr.fit_transform(X_train)
X_test_scaled_lr = scaler_lr.transform(X_test)
print("Features scaled for Logistic Regression.")

# Add polynomial features
# Adjust degree based on problem - starting with degree 2
poly = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False to avoid adding a column of ones
X_train_poly = poly.fit_transform(X_train_scaled_lr)
X_test_poly = poly.transform(X_test_scaled_lr)
print(f"Added polynomial features. New training shape: {X_train_poly.shape}")


# Hyperparameter Tuning using RandomizedSearchCV
print("\nPerforming RandomizedSearchCV for Hyperparameter Tuning...")
# Optimize the C parameter using RandomizedSearchCV
# Using a log-uniform distribution for C is more suitable for scale-invariant search
param_dist = {'C': loguniform(1e-3, 1e3)}

# Using the data with polynomial features
# n_iter controls how many parameter combinations are sampled
random_search = RandomizedSearchCV(
    LogisticRegression(random_state=42, penalty='l2'),
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings that are sampled. Adjust as needed.
    scoring='accuracy',
    random_state=42,
    n_jobs=-1  # Use all available cores
)
random_search.fit(X_train_poly, y_train)

best_C = random_search.best_params_['C']
print(f"Best C found by RandomizedSearchCV: {best_C}")

# Define the Logistic Regression model with the best C found
model_lr = LogisticRegression(random_state=42, C=best_C, penalty='l2', solver='lbfgs', max_iter=2000)

# Train the model on the data with polynomial features
print("\nTraining Logistic Regression Model with best C and polynomial features...")
model_lr.fit(X_train_poly, y_train)
print("Model training complete.")

# Evaluate the model on the test data with polynomial features
print("Evaluating Logistic Regression Model...")
y_pred_lr = model_lr.predict(X_test_poly)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

print(f"\nAccuracy (Logistic Regression): {accuracy_lr:.4f}")
print(f"Precision (Logistic Regression): {precision_lr:.4f}")
print(f"Recall (Logistic Regression): {recall_lr:.4f}")
print(f"F1-Score (Logistic Regression): {f1_lr:.4f}")

print("\nConfusion Matrix (Logistic Regression):")
print(conf_matrix_lr)

print("\n--- Confusion Matrix Breakdown (Logistic Regression) ---")
TN_lr, FP_lr, FN_lr, TP_lr = conf_matrix_lr.ravel()
print(f"True Negatives (Legitimate Correctly Classified): {TN_lr}")
print(f"False Positives (Legitimate Classified as Phishing): {FP_lr}")
print(f"False Negatives (Phishing Classified as Legitimate): {FN_lr}")
print(f"True Positives (Phishing Correctly Classified): {TP_lr}")