import numpy as np
import pandas as pd
import pickle
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Example data loading (replace this with your actual data loading)
# Load dataset
df = pd.read_csv('creditcard.csv')

df_class_0 = df[df['Class'] == 0]   # Non fraud records (The majority class)
df_class_1 = df[df['Class'] == 1]
# Data preprocessing
columns = "V1 V2 V3 V4 V5 V6 V7 V8 V9 V10 V11 V12 V13 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23 V24 V25 V26 V27 V28 Amount".split()
X = df[columns].values
Y = df['Class'].values
df_class_1 = df_class_1.sample(df_class_0.shape[0], replace=True)
df_class_1.shape
df['Class'].value_counts()

# 1. Split the dataset into a temporary training set and the test set
D_train_temp, D_test = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)

# 2. Split the temporary training set into the final training set and the validation set
D_train, T_valid   = train_test_split(D_train_temp, test_size=0.25, stratify=D_train_temp['Class'], random_state=42)

x_train = D_train.iloc[:, :-1]
y_train = D_train.iloc[:, -1]

x_test = D_test.iloc[:, :-1]
y_test = D_test.iloc[:, -1]

x_val = T_valid.iloc[:, :-1]
y_val = T_valid.iloc[:, -1]
# Combine training and validation sets for training
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Combine training and validation sets for training
x_combined = np.concatenate((x_train, x_val))
y_combined = np.concatenate((y_train, y_val))

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, solver='saga')
lr_model.fit(x_combined, y_combined)

# Save the model to a file
with open('Linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Evaluate the model
y_pred = lr_model.predict(x_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

