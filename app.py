import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('admission_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Ivy League Admission Predictor 🎓")

# Input fields
gre = st.number_input(
    "GRE Score",
    min_value=260,
    max_value=340,
    help="Enter GRE score (Min: 260, Max: 340)"
)

toefl = st.number_input(
    "TOEFL Score",
    min_value=0,
    max_value=120,
    help="Enter TOEFL score (Min: 0, Max: 120)"
)

univ_rating = st.slider("University Rating", 1, 5)
sop = st.slider("SOP Strength (0-5)", 0.0, 5.0, step=0.5)
lor = st.slider("LOR Strength (0-5)", 0.0, 5.0, step=0.5)
cgpa = st.number_input("CGPA (0-10)", 0.0, 10.0)
research = st.selectbox("Research Experience", ['No', 'Yes'])

# Convert research to binary
research = 1 if research == 'Yes' else 0

# Predict button
if st.button("Predict Admission Chance"):
    features = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, research]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    st.success(f"Predicted Chance of Admission: {prediction[0]*100:.2f}%")

