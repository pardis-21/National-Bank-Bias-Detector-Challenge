import streamlit as st
import pandas as pd
from bias_engine import detect_overtrading  # Importing your backend logic
from chatbot import get_ai_advice           # Importing your AI logic

st.title("National Bank Bias Detector")

# 1. UI for Uploading
file = st.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
    
    # 2. Call the Backend
    overtrading_results = detect_overtrading(df)
    
    # 3. Display it
    st.write(overtrading_results)