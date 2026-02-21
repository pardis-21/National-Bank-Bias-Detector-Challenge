import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBC Bias Detector", layout="wide")
st.title("🏦 National Bank: AI Bias Detector")

# --- API SETUP ---
# Get your free key from: https://aistudio.google.com/
# Store it in .streamlit/secrets.toml as GEMINI_API_KEY = "your_key"
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except:
    st.error("Please set your GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# --- SIDEBAR: MULTI-FILE UPLOAD ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload all Trading CSVs", 
        type="csv", 
        accept_multiple_files=True
    )

# --- DATA PROCESSING ---
if uploaded_files:
    # Merge multiple CSVs
    dfs = []
    for file in uploaded_files:
        temp_df = pd.read_csv(file)
        dfs.append(temp_df)
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # --- BIAS LOGIC CALCULATIONS ---
    
    # 1. Overtrading (Trades per Hour)
    df['hour'] = df['timestamp'].dt.hour
    hourly_trades = df.groupby('hour').size()
    max_trades_hour = hourly_trades.idxmax()
    is_overtrading = hourly_trades.max() > 10 # Example threshold

    # 2. Loss Aversion (Holding losers longer than winners)
    # Assume we calculate hold time as (exit_price - entry_price) duration if not provided
    avg_win = df[df['profit_loss'] > 0]['profit_loss'].mean()
    avg_loss = abs(df[df['profit_loss'] < 0]['profit_loss'].mean())
    loss_ratio = avg_loss / avg_win if avg_win > 0 else 0

    # 3. Revenge Trading (Large trade right after a loss)
    df['prev_PL'] = df['profit_loss'].shift(1)
    df['prev_time'] = df['timestamp'].shift(1)
    df['time_diff'] = (df['timestamp'] - df['prev_time']).dt.total_seconds() / 60
    
    # Flag if trade > avg_size and happens < 15 mins after a loss
    revenge_trades = df[(df['prev_PL'] < 0) & (df['time_diff'] < 15)]
    has_revenge_bias = not revenge_trades.empty

    # --- DASHBOARD UI ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", len(df))
    col2.metric("Loss Ratio", f"{loss_ratio:.2f}x")
    col3.metric("Peak Trading Hour", f"{max_trades_hour}:00")

    # Visualizations
    st.subheader("Performance Timeline")
    fig = px.line(df, x='timestamp', y='balance', title="Account Balance Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # --- AI CHATBOT (THE COACH) ---
    st.divider()
    st.subheader("💬 AI Trading Coach")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your biases..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build context for the AI
        bias_summary = f"""
        User Stats:
        - Total Trades: {len(df)}
        - Overtrading detected: {is_overtrading} (Peak at {max_trades_hour}:00)
        - Loss Ratio: {loss_ratio:.2f} (Target < 1.0)
        - Revenge Trading detected: {has_revenge_bias}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash", # Latest free model
            contents=f"You are a National Bank trading coach. Context: {bias_summary}. User asks: {prompt}"
        )

        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

else:
    st.info("Waiting for CSV uploads to begin analysis...")