import streamlit as st
import tensorflow  as tf # type: ignore
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scale.pkl')
label_encoder = joblib.load('label_encode.pkl')

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="milkquality.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Milk Quality")
st.write("Menentukan kualitas Susu.")

# Form input pengguna
P = st.number_input("Kandungan pH (pH)", min_value=0.0, max_value=10.0, value=5.0)
T = st.number_input("Kandungan Temperature (P)", min_value=0, max_value=70, value=30)
taste = st.number_input("Kandungan Taste (K)", min_value=0, max_value=1, value=0)
odor = st.number_input("Odor (°C)", min_value=0, max_value=1, value=0)
fat = st.number_input("Fat (%)", min_value=0, max_value=1, value=0)
turbidity = st.number_input("Turbidity", min_value=0, max_value=1, value=0)
colour = st.number_input("Colour (mm)", min_value=250, max_value=270, value=255)

if st.button("Hasil Kualitas Susu"):
    # Preprocessing input
    input_data = np.array([[P, T, taste, odor, fat,turbidity,colour]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_label = np.argmax(prediction)
    crop_name = label_encoder.inverse_transform([predicted_label])[0]

    st.success(f"Kualitas Susu: **{crop_name.upper()}**")
