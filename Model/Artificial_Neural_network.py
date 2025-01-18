import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv('creditcard.csv')

# Balance the dataset by oversampling the minority class
df_class_0 = df[df['Class'] == 0]  # Not fraud records
df_class_1 = df[df['Class'] == 1]
df_class_1 = df_class_1.sample(df_class_0.shape[0], replace=True)
df = pd.concat([df_class_0, df_class_1], axis=0)

# Split the dataset into a temporary training set and the test set
D_train_temp, D_test = train_test_split(df, test_size=0.2, stratify=df['Class'], random_state=42)

# Split the temporary training set into the final training set and the validation set
D_train, T_valid = train_test_split(D_train_temp, test_size=0.25, stratify=D_train_temp['Class'], random_state=42)

# Extract features and labels
x_train = D_train.iloc[:, :-1]
y_train = D_train.iloc[:, -1]

x_test = D_test.iloc[:, :-1]
y_test = D_test.iloc[:, -1]

x_val = T_valid.iloc[:, :-1]
y_val = T_valid.iloc[:, -1]

# Define the MLP model with best parameters
best_mlp = MLPClassifier(
    hidden_layer_sizes=(40, 20, 10, 5),
    activation='relu',
    solver='sgd',
    alpha=0.05,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    random_state=42
)

# Train the model
best_mlp.fit(np.vstack((x_train, x_val)), np.hstack((y_train, y_val)))

print("\nDevelopment set performance:")
test_report = classification_report(y_test, best_mlp.predict(x_test))
print(test_report)

# Save the model to a file
with open('Artificial_Neural_network.pkl', 'wb') as f:
    pickle.dump(best_mlp, f)
