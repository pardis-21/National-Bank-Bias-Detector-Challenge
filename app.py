import streamlit as st
import pandas as pd

st.title("National Bank: AI Bias Detector for Traders Hackathon Challenge")

# 1. File Upload
uploaded_file = st.file_uploader("Upload Trading History (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Loaded Successfully!")
    
    # 2. Logic for "Revenge Trading"
    # (Checking if a large trade follows a loss)
    
    # 3. AI Chatbot Sidebar
    with st.sidebar:
        st.header("AI Trading Coach")
        user_input = st.text_input("Ask about your biases:")