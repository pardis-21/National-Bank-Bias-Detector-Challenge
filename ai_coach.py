from google import genai
import streamlit as st

def get_chatbot_response(bias_summary, user_prompt):
    """
    Handles the connection to Gemini and returns the coach's advice.
    """
    try:
        # 1. Setup Client
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        # 2. System Instruction
        system_prompt = (
            "You are a professional trading coach at National Bank. "
            "Use the following bias report to give the trader specific, "
            "actionable advice. Be empathetic but data-driven."
        )

        # 3. Call Model (Using the corrected model name)
        response = client.models.generate_content(
            model = "gemini-2.0-flash-lite",
            contents=f"{system_prompt}\n\nBias Report:\n{bias_summary}\n\nTrader asks: {user_prompt}",
        )
        
        return response.text

    except Exception as e:
        return f"❌ AI Error: {str(e)}"