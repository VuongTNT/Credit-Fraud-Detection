from flask import Flask, request, render_template
import pandas as pd
import pickle
from keras.models import load_model

app = Flask(__name__)

# Load the models and scaler
with open('Linear_regression.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('Decision_tree.pkl', 'rb') as f:
    dt_model = pickle.load(f)

with open('Random_Forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('SVM.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('Artificial_Neural_network.pkl', 'rb') as f:
    ann_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data_string = request.form['data']  # Get comma-separated data
    selected_model = request.form['model']

    # Split the data string into a list of floats
    try:
        data_list = [float(x) for x in data_string.split(',')]
    except ValueError:
        return render_template('error.html', error_message="Invalid data format. Please use comma-separated numbers.")

    # Ensure the data list has the expected length (30 features)
    if len(data_list) != 30:
        return render_template('error.html', error_message="Incorrect number of values. Please provide all 30 features (Time, V1, V2, ..., Amount).")

    # Create a dictionary with data in the correct order based on your CSV
    input_data = {
        'Time': data_list[0],
        **{f'V{i+1}': data_list[i+1] for i in range(28)},  # Unpack V features
        'Amount': data_list[29]
    }

    # Convert to DataFrame
    input_df_full = pd.DataFrame([input_data])
    input_df = input_df_full.drop(columns=['Time'])  # Drop 'Time' column for scaling

    # Scale the data
    input_df_scaled = scaler.transform(input_df)

    # Re-add the 'Time' column after scaling
    input_df_scaled = pd.DataFrame(input_df_scaled, columns=input_df.columns)
    input_df_scaled.insert(0, 'Time', input_df_full['Time'].values)

    if selected_model == 'Linear Regression':
        y_pred = lr_model.predict(input_df_full)
        is_fraud = y_pred[0]
        fraud_prob = y_pred[0] * 100  # Placeholder, Linear Regression does not provide probability by default

        return render_template('predict.html', fraud_prob=fraud_prob, is_fraud=is_fraud)

    elif selected_model == 'Decision Tree':
        y_pred = dt_model.predict(input_df_full)
        is_fraud = y_pred[0]
        fraud_prob = y_pred[0] * 100  # Placeholder, Decision Tree does not provide probability by default

        return render_template('predict.html', fraud_prob=fraud_prob, is_fraud=is_fraud)

    elif selected_model == 'Random Forest':
        y_pred_prob = rf_model.predict_proba(input_df_full)[0]
        is_fraud = (y_pred_prob[1] >= 0.5).astype(int)
        fraud_prob = y_pred_prob[1] * 100

        return render_template('predict.html', fraud_prob=fraud_prob, is_fraud=is_fraud)

    elif selected_model == 'SVM':
        y_pred = svm_model.predict(input_df_full)
        is_fraud = y_pred[0]
        fraud_prob = y_pred[0] * 100  # Placeholder, SVM does not provide probability by default

        return render_template('predict.html', fraud_prob=fraud_prob, is_fraud=is_fraud)

    elif selected_model == 'Artificial Neural Network':
        y_pred_prob = ann_model.predict_proba(input_df_scaled)[0]
        is_fraud = (y_pred_prob[1] >= 0.5).astype(int)
        fraud_prob = y_pred_prob[1] * 100

        return render_template('predict.html', fraud_prob=fraud_prob, is_fraud=is_fraud)

    else:
        return render_template('error.html', error_message="Invalid model selection.")

if __name__ == '__main__':
    app.run(debug=True)
