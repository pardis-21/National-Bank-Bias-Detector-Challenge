import streamlit as st
import pandas as pd

# Multi-file uploader
uploaded_files = st.file_uploader("Upload all 20 Trading CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Combine all files into one
    list_of_dfs = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(list_of_dfs, ignore_index=True)
    
    # Ensure timestamps are actual dates for analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    st.success(f"Successfully merged {len(uploaded_files)} files! Total trades: {len(df)}")
    st.dataframe(df.head()) # Preview the data
    
    
    # Initialize chat history in session state so it doesn't disappear on refresh
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask your AI Trading Coach..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # PLACEHOLDER FOR ML LOGIC:
    # You would send the 'df' analysis results to an LLM here
    response = f"AI Coach: I see you've made {len(df)} trades. Let's look at your overtrading bias."
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})