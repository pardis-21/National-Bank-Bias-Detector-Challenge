import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# ── Import our bias engine ────────────────────────────────────
from bias_engine import run_all, detect_overtrading, detect_loss_aversion, detect_revenge_trading

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBC Bias Detector", layout="wide")
st.title("🏦 National Bank: AI Bias Detector")

# ─────────────────────────────────────────────────────────────
# API SETUP
# Get your free key from: https://aistudio.google.com/
# Store it in .streamlit/secrets.toml as GEMINI_API_KEY = "your_key"
# ─────────────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client  = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Please set your GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload all Trading CSVs",
        type="csv",
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2. Detection Thresholds")
    max_per_hour   = st.number_input("Max trades/hour (Overtrading)",      min_value=1,   value=10)
    max_vol_ratio  = st.number_input("Max volume/balance ratio",            min_value=0.5, value=3.0, step=0.5)
    loss_win_ratio = st.number_input("Loss/Win ratio (Loss Aversion)",      min_value=1.0, value=1.5, step=0.1)
    revenge_mult   = st.number_input("Revenge trade size multiplier",       min_value=1.1, value=1.5, step=0.1)
    revenge_time   = st.number_input("Revenge time window (minutes)",       min_value=1,   value=15)

    st.divider()
    st.caption("Required CSV columns: `timestamp`, `buy_sell`, `asset`, `quantity`, `entry_price`, `exit_price`, `profit_loss`, `balance`")

# ─────────────────────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────────────────────
if not uploaded_files:
    st.info("⬅️ Upload one or more CSV files in the sidebar to begin analysis.")
    st.stop()

dfs = [pd.read_csv(f) for f in uploaded_files]
df  = pd.concat(dfs, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# RUN BIAS ENGINE
# ─────────────────────────────────────────────────────────────
biases = run_all(
    df,
    max_per_hour     = int(max_per_hour),
    max_vol_ratio    = float(max_vol_ratio),
    loss_win_ratio   = float(loss_win_ratio),
    revenge_mult     = float(revenge_mult),
    revenge_time_min = int(revenge_time),
)

ot = biases["overtrading"]
la = biases["loss_aversion"]
rt = biases["revenge_trading"]

# ─────────────────────────────────────────────────────────────
# QUICK STATS (original metrics + bias flags)
# ─────────────────────────────────────────────────────────────
avg_win  = df[df["profit_loss"] > 0]["profit_loss"].mean() if (df["profit_loss"] > 0).any() else 0
avg_loss = df[df["profit_loss"] < 0]["profit_loss"].abs().mean() if (df["profit_loss"] < 0).any() else 0
loss_ratio = round(avg_loss / avg_win, 2) if avg_win > 0 else 0

peak_hour = df.set_index("timestamp").resample("1h").size().idxmax()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades",     len(df))
col2.metric("Loss Ratio",       f"{loss_ratio:.2f}x")
col3.metric("Peak Trading Hour", f"{peak_hour.strftime('%H:%M')}")
col4.metric("Revenge Trades",   rt["details"].get("revenge_trade_count", 0),
            delta="⚠ Detected" if rt["flagged"] else "✓ Clear",
            delta_color="inverse" if rt["flagged"] else "normal")
col5.metric("Avg Win / Loss",   f"${avg_win:.0f} / ${avg_loss:.0f}")

# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
st.subheader("Performance Timeline")
fig_balance = px.line(df, x="timestamp", y="balance", title="Account Balance Over Time")
fig_balance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_balance, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    colors = ["green" if v >= 0 else "red" for v in df["profit_loss"]]
    fig_pnl = go.Figure(go.Bar(x=df["timestamp"], y=df["profit_loss"], marker_color=colors))
    fig_pnl.update_layout(title="P/L Per Trade", xaxis_title="Time", yaxis_title="P/L")
    st.plotly_chart(fig_pnl, use_container_width=True)

with c2:
    hourly_counts = df.set_index("timestamp").resample("1h").size().reset_index(name="trades")
    fig_hourly = px.bar(hourly_counts, x="timestamp", y="trades", title="Trades Per Hour",
                        color="trades", color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig_hourly, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# BIAS DETECTION CARDS
# ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Bias Detection Results")

def bias_card(title: str, result: dict, description: str):
    flagged = result["flagged"]
    icon    = "🚨" if flagged else "✅"
    status  = "DETECTED" if flagged else "CLEAR"
    color   = "#ffcccc" if flagged else "#ccffcc"
    border  = "#cc0000" if flagged else "#007700"

    reasons_html = "".join(f"<li>{r}</li>" for r in result["reasons"]) if result["reasons"] else "<li>No issues found.</li>"
    details_html = " &nbsp;|&nbsp; ".join(
        f"<b>{k.replace('_',' ').title()}</b>: {v}"
        for k, v in result["details"].items()
        if k != "note"
    )

    st.markdown(f"""
    <div style="border:2px solid {border};border-radius:10px;padding:1rem;background:{color};margin-bottom:1rem;">
        <h4 style="margin:0 0 0.3rem 0;">{icon} {title} &mdash; <span style="color:{border}">{status}</span></h4>
        <p style="margin:0 0 0.5rem 0;color:#444;font-size:0.9rem;">{description}</p>
        <ul style="margin:0 0 0.5rem 0;">{"".join(f"<li>{r}</li>" for r in result["reasons"]) if result["reasons"] else "<li>No issues found.</li>"}</ul>
        <p style="margin:0;font-size:0.8rem;color:#555;">{details_html}</p>
    </div>
    """, unsafe_allow_html=True)

bias_card(
    "🔄 Overtrading",
    ot,
    "Trading too frequently — bursts within single hours, high volume vs balance, or rapid position flipping.",
)
bias_card(
    "😰 Loss Aversion",
    la,
    "Holding losing trades too long while cutting winners short.",
)
bias_card(
    "😤 Revenge Trading",
    rt,
    "Opening oversized positions shortly after a loss to 'win back' money.",
)

# Revenge trade detail table
if rt["flagged_trades"]:
    st.markdown("**Flagged Revenge Trades:**")
    st.dataframe(pd.DataFrame(rt["flagged_trades"]), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# AI CHATBOT COACH
# ─────────────────────────────────────────────────────────────
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

    # ── Build rich bias context for Gemini ───────────────────
    bias_summary = f"""
Trader Performance Summary:
- Total Trades: {len(df)}
- Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}
- Net P/L: {df['profit_loss'].sum():.2f}
- Average Win: ${avg_win:.2f}  |  Average Loss: ${avg_loss:.2f}
- Loss Ratio: {loss_ratio:.2f}x (target < 1.0)
- Peak Trading Hour: {peak_hour.strftime('%H:%M')}

Bias Detection Results:
1. Overtrading — {'DETECTED' if ot['flagged'] else 'CLEAR'}
   Reasons: {'; '.join(ot['reasons']) if ot['reasons'] else 'None'}
   Details: {ot['details']}

2. Loss Aversion — {'DETECTED' if la['flagged'] else 'CLEAR'}
   Reasons: {'; '.join(la['reasons']) if la['reasons'] else 'None'}
   Details: {la['details']}

3. Revenge Trading — {'DETECTED' if rt['flagged'] else 'CLEAR'}
   Reasons: {'; '.join(rt['reasons']) if rt['reasons'] else 'None'}
   Flagged trades count: {rt['details'].get('revenge_trade_count', 0)}
"""

    system_prompt = (
        "You are a professional trading coach at National Bank. "
        "You have access to a trader's bias analysis report. "
        "Be empathetic, specific, and always reference the actual numbers from the report. "
        "Give concrete, actionable advice. Never give generic tips."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{system_prompt}\n\nBias Report:\n{bias_summary}\n\nTrader asks: {prompt}",
    )

    with st.chat_message("assistant"):
        st.markdown(response.text)

    st.session_state.messages.append({"role": "assistant", "content": response.text})