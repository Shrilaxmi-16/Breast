# app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (you can replace this with your cancer dataset)
@st.cache
def load_data():
    data = pd.read_csv('data.csv')  # Update with your actual dataset
    return data

# Build RandomForestClassifier model
def build_model():
    data = load_data()
    X = data.drop(columns=['cancer_type'])  # 'cancer_type' is your target column
    y = data['cancer_type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy

st.title("Cancer Type Prediction App")
st.write("Upload your patient data below:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write(input_data)

    # Build the model and predict
    model, accuracy = build_model()
    st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    
    prediction = model.predict(input_data)
    st.write("Predicted Cancer Types:")
    st.write(prediction)
