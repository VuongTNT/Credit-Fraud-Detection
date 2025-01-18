import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import random as rd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
data = pd.read_csv('creditcard.csv')

# Transform data
scaler = RobustScaler()
columns = "Time V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 Amount".split()
for column in columns:
    data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1))

# Undersampling
fraud_data = data[data['Class'] == 1]
non_fraud_data = data[data['Class'] == 0].sample(n=len(fraud_data), random_state=42)
balanced_data = pd.concat([fraud_data, non_fraud_data])

# Remove outliers from the balanced dataset
for column in columns:
    fraud = balanced_data[column][balanced_data['Class'] == 1].values
    q25, q75 = np.percentile(fraud, 25), np.percentile(fraud, 75)
    iqr = q75 - q25
    cut_off = iqr * 5
    lower, upper = q25 - cut_off, q75 + cut_off
    balanced_data = balanced_data[~((balanced_data[column] < lower) | (balanced_data[column] > upper))]

# Prepare features and labels
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=565)

# Initialize the SVM model with best parameters
svc = SVC(C=0.5, kernel='linear', probability=True)

# Train the model
svc.fit(X_train, y_train)

# Evaluate the model
y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model to a pickle file
with open('SVM.pkl', 'wb') as f:
    pickle.dump(svc, f)

print("Model saved to SVM.pkl")
