import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model and the scaler used during training
model = joblib.load(r'C:\Users\admin\Documents\Projects\Credit Card Fraud Detection Project\fraud_detection_model.pkl')
scaler = joblib.load(r'C:\Users\admin\Documents\Projects\Credit Card Fraud Detection Project\scaler.pkl')

# Streamlit UI setup
st.title("Credit Card Fraud Detection")
st.write("""
This web app will help you predict whether a given transaction is fraudulent or not.
Enter the transaction details below.
""")

# Input fields for the user to provide transaction details
transaction_amount = st.number_input('Transaction Amount', min_value=0.0, value=100.0, step=0.01)
time = st.number_input('Transaction Time (in seconds)', min_value=0, value=1000, step=1)

# Adding inputs for 'V1' to 'V30'
v1 = st.number_input('V1', min_value=-10.0, value=0.0, step=0.1)
v2 = st.number_input('V2', min_value=-10.0, value=0.0, step=0.1)
v3 = st.number_input('V3', min_value=-10.0, value=0.0, step=0.1)
v4 = st.number_input('V4', min_value=-10.0, value=0.0, step=0.1)
v5 = st.number_input('V5', min_value=-10.0, value=0.0, step=0.1)
v6 = st.number_input('V6', min_value=-10.0, value=0.0, step=0.1)
v7 = st.number_input('V7', min_value=-10.0, value=0.0, step=0.1)
v8 = st.number_input('V8', min_value=-10.0, value=0.0, step=0.1)
v9 = st.number_input('V9', min_value=-10.0, value=0.0, step=0.1)
v10 = st.number_input('V10', min_value=-10.0, value=0.0, step=0.1)

# Adding more 'V' variables for V11 to V30
v11 = st.number_input('V11', min_value=-10.0, value=0.0, step=0.1)
v12 = st.number_input('V12', min_value=-10.0, value=0.0, step=0.1)
v13 = st.number_input('V13', min_value=-10.0, value=0.0, step=0.1)
v14 = st.number_input('V14', min_value=-10.0, value=0.0, step=0.1)
v15 = st.number_input('V15', min_value=-10.0, value=0.0, step=0.1)
v16 = st.number_input('V16', min_value=-10.0, value=0.0, step=0.1)
v17 = st.number_input('V17', min_value=-10.0, value=0.0, step=0.1)
v18 = st.number_input('V18', min_value=-10.0, value=0.0, step=0.1)
v19 = st.number_input('V19', min_value=-10.0, value=0.0, step=0.1)
v20 = st.number_input('V20', min_value=-10.0, value=0.0, step=0.1)
v21 = st.number_input('V21', min_value=-10.0, value=0.0, step=0.1)
v22 = st.number_input('V22', min_value=-10.0, value=0.0, step=0.1)
v23 = st.number_input('V23', min_value=-10.0, value=0.0, step=0.1)
v24 = st.number_input('V24', min_value=-10.0, value=0.0, step=0.1)
v25 = st.number_input('V25', min_value=-10.0, value=0.0, step=0.1)
v26 = st.number_input('V26', min_value=-10.0, value=0.0, step=0.1)
v27 = st.number_input('V27', min_value=-10.0, value=0.0, step=0.1)
v28 = st.number_input('V28', min_value=-10.0, value=0.0, step=0.1)
v29 = st.number_input('V29', min_value=-10.0, value=0.0, step=0.1)
v30 = st.number_input('V30', min_value=-10.0, value=0.0, step=0.1)

# When the user submits, predict the fraud
if st.button('Predict'):
    # Creating an input dataframe from user input
    user_data = pd.DataFrame({
        'Amount': [transaction_amount],
        'Time': [time],
        'V1': [v1],
        'V2': [v2],
        'V3': [v3],
        'V4': [v4],
        'V5': [v5],
        'V6': [v6],
        'V7': [v7],
        'V8': [v8],
        'V9': [v9],
        'V10': [v10],
        'V11': [v11],
        'V12': [v12],
        'V13': [v13],
        'V14': [v14],
        'V15': [v15],
        'V16': [v16],
        'V17': [v17],
        'V18': [v18],
        'V19': [v19],
        'V20': [v20],
        'V21': [v21],
        'V22': [v22],
        'V23': [v23],
        'V24': [v24],
        'V25': [v25],
        'V26': [v26],
        'V27': [v27],
        'V28': [v28],
        'V29': [v29],
        'V30': [v30]
    })

    # Ensure that the user data matches the model's expected features (30 features)
    if user_data.shape[1] == 30:
        # Scale the input data
        user_data_scaled = scaler.transform(user_data)

        # Predict whether the transaction is fraudulent or not
        prediction = model.predict(user_data_scaled)
        
        if prediction[0] == 1:
            st.error('This transaction is likely to be FRAUDULENT.')
        else:
            st.success('This transaction is likely to be LEGITIMATE.')
    else:
        st.error("The input data doesn't match the required number of features. Please check and provide all 30 features.")

