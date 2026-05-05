
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and label encoder
try:
    mlp_model = joblib.load('mlp_model.joblib')
    loaded_label_encoder = joblib.load('label_encoder.joblib')
    # The classes_ attribute of LabelEncoder stores the original class labels
    species_names = loaded_label_encoder.classes_
except FileNotFoundError:
    st.error("Error: Model or label encoder files not found. Please ensure 'mlp_model.joblib' and 'label_encoder.joblib' are in the same directory.")
    st.stop()

st.title("Iris Species Predictor")
st.write("Enter the features of the Iris flower to predict its species.")

# Input fields for features
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Use a button to trigger prediction
if st.button("Predict Species"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

    # Make prediction
    prediction_numeric = mlp_model.predict(input_data)
    prediction_species = loaded_label_encoder.inverse_transform(prediction_numeric)

    st.success(f"Predicted Iris Species: **{prediction_species[0]}**")

