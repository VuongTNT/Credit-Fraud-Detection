from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

dtree = DecisionTreeClassifier()

df = pd.read_csv('creditcard.csv')

#dataset informations
df.info()

df_class_0 = df[df['Class'] == 0]  # Not fraud records
df_class_1 = df[df['Class'] == 1]

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
# Train the model
dtree.fit(np.concatenate((x_train, x_val)), np.concatenate((y_train, y_val)))
print(classification_report(y_test, dtree.predict(x_test), target_names=['Not fraud', 'Fraud']))

# Save the model
with open('Decision_tree.pkl', 'wb') as f:
    pickle.dump(dtree, f)
