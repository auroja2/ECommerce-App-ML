import streamlit as st
import pickle
import numpy as np
import os

# Set up paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = r"C:\Users\auroj\Project4\app\models\scaler.pkl"
model_path = r"C:\Users\auroj\Project4\app\models\model.pkl"


# Load the scaler and model
try:
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

except FileNotFoundError:
    st.error("Model or scaler file not found! Please check the paths.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 🎨 Streamlit UI Design
st.set_page_config(page_title="E-Commerce Sales Predictor", layout="centered")

# Header with an image
st.image("https://source.unsplash.com/800x300/?ecommerce,shopping", use_column_width=True)
st.title("🛍️ E-Commerce Sales Predictor")
st.markdown("### Predict customer sales based on their activity.")

# Sidebar for additional details
with st.sidebar:
    st.header("About the Model")
    st.markdown("""
    - **Input Features**:  
      - Average Session Length  
      - Time Spent on App  
      - Length of Membership  
    - **Technology Used**:  
      - Streamlit  
      - Scikit-learn (for Scaling & Model)  
      - NumPy  
    """)

# 📌 Input Fields for User Data
st.subheader("Enter Customer Data")

col1, col2 = st.columns(2)
with col1:
    avg_session_length = st.number_input("📏 Avg. Session Length", min_value=0.0, step=0.1)
    length_of_membership = st.number_input("🎟️ Length of Membership", min_value=0.0, step=0.1)
with col2:
    time_on_app = st.number_input("📱 Time on App", min_value=0.0, step=0.1)

# 🚀 Prediction Button
if st.button("🔮 Predict Sales"):
    try:
        # Prepare and scale the input data
        input_data = np.array([[avg_session_length, time_on_app, length_of_membership]])
        scaled_data = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(scaled_data)[0]

        # Display the result
        st.success(f"💰 Predicted Sales: **${prediction:.2f}**")

        # 🎨 Visual Enhancement
        st.markdown("""
        <style>
        .success { font-size: 20px; font-weight: bold; color: green; }
        </style>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in prediction: {e}")



