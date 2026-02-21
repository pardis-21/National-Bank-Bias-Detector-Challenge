from ai_coach import get_chatbot_response
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import re
from google import genai
from google.genai import errors as genai_errors
from bias_engine import run_all

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBC Bias Detector", layout="wide")

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
    client = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR — navigation + settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 NBC Bias Detector")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "📚 Learning Centre"],
        label_visibility="collapsed",
    )

    st.divider()

    if page == "📊 Dashboard":
        st.header("1. Upload Data")
        uploaded_files = st.file_uploader(
            "Upload all Trading CSVs",
            type="csv",
            accept_multiple_files=True,
        )

        st.divider()
        st.header("2. Detection Thresholds")
        # max_per_hour is inserted dynamically below after data loads
        max_vol_ratio  = st.number_input("Max volume/balance ratio",       min_value=0.5, value=3.0,  step=0.5)
        loss_win_ratio = st.number_input("Loss/Win ratio (Loss Aversion)",  min_value=1.0, value=1.5,  step=0.1)
        revenge_mult   = st.number_input("Revenge trade size multiplier",   min_value=1.1, value=1.5,  step=0.1)
        revenge_time   = st.number_input("Revenge time window (minutes)",   min_value=1,   value=15,   step=1)

        st.divider()
        st.header("3. Gemini Model")
        model_choice = st.selectbox(
            "Model",
            ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash-8b"],
            index=0,
            help="gemini-2.0-flash-lite has the most generous free-tier quota.",
            )

        st.divider()
        st.caption("Required columns: `timestamp` `buy_sell` `asset` `quantity` `entry_price` `exit_price` `profit_loss` `balance`")


# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("🏦 National Bank: AI Bias Detector")

    # ── Load & cache data ────────────────────────────────────
    if uploaded_files:
        file_fingerprint = [(f.name, f.size) for f in uploaded_files]
        if st.session_state.get("file_fingerprint") != file_fingerprint:
            dfs    = [pd.read_csv(f) for f in uploaded_files]
            df_raw = pd.concat(dfs, ignore_index=True)
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
            st.session_state["df"]               = df_raw
            st.session_state["file_fingerprint"] = file_fingerprint

    if "df" not in st.session_state:
        st.info("⬅️ Upload one or more CSV files in the sidebar to begin analysis.")
        st.stop()

    df = st.session_state["df"]

    # ── Dynamic threshold calculation ────────────────────────
    # Calculate avg trades/hour across their whole history,
    # suggest 2× that average as the threshold (minimum 15)
    total_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
    if total_hours < 1:
        total_hours = 1
    avg_hourly_volume  = len(df) / total_hours
    dynamic_suggestion = max(int(avg_hourly_volume * 2), 15)

    max_per_hour = st.sidebar.number_input(
        "Max trades/hour (Overtrading)",
        min_value=1,
        value=dynamic_suggestion,
        help=f"Auto-set to 2× your avg hourly rate ({avg_hourly_volume:.1f} trades/hr). Adjust as needed.",
    )

    # ── Run bias engine ──────────────────────────────────────
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

    # ── Quick stats ──────────────────────────────────────────
    avg_win    = df[df["profit_loss"] > 0]["profit_loss"].mean() if (df["profit_loss"] > 0).any() else 0
    avg_loss   = df[df["profit_loss"] < 0]["profit_loss"].abs().mean() if (df["profit_loss"] < 0).any() else 0
    loss_ratio = round(avg_loss / avg_win, 2) if avg_win > 0 else 0
    peak_hour  = df.set_index("timestamp").resample("1h").size().idxmax()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Trades",      len(df))
    col2.metric("Loss Ratio",        f"{loss_ratio:.2f}x")
    col3.metric("Peak Trading Hour", peak_hour.strftime("%H:%M"))
    col4.metric("Revenge Trades",    rt["details"].get("revenge_trade_count", 0),
                delta="⚠ Detected" if rt["flagged"] else "✓ Clear",
                delta_color="inverse" if rt["flagged"] else "normal")
    col5.metric("Avg Win / Loss",    f"${avg_win:.0f} / ${avg_loss:.0f}")

    # ── Charts ───────────────────────────────────────────────
    st.subheader("Performance Timeline")
    fig_balance = px.line(df, x="timestamp", y="balance", title="Account Balance Over Time")
    fig_balance.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_balance, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        colors  = ["green" if v >= 0 else "red" for v in df["profit_loss"]]
        fig_pnl = go.Figure(go.Bar(x=df["timestamp"], y=df["profit_loss"], marker_color=colors))
        fig_pnl.update_layout(title="P/L Per Trade", xaxis_title="Time", yaxis_title="P/L")
        st.plotly_chart(fig_pnl, use_container_width=True)

    with c2:
        hourly_counts = df.set_index("timestamp").resample("1h").size().reset_index(name="trades")
        current_max   = hourly_counts["trades"].max()
        chart_ceiling = max(current_max, int(max_per_hour), 100)

        fig_hourly = px.bar(
            hourly_counts,
            x="timestamp",
            y="trades",
            title="Trades Per Hour",
            color="trades",
            range_color=[0, int(max_per_hour)],  # red starts exactly at threshold
            color_continuous_scale="RdYlGn_r",
            range_y=[0, chart_ceiling],
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    # ── Bias detection cards ─────────────────────────────────
    st.divider()
    st.subheader("🔍 Bias Detection Results")

    def bias_card(title, result, description):
        flagged      = result["flagged"]
        icon         = "🚨" if flagged else "✅"
        status       = "DETECTED" if flagged else "CLEAR"
        color        = "#ffcccc" if flagged else "#ccffcc"
        border       = "#cc0000" if flagged else "#007700"
        reasons_html = "".join(f"<li>{r}</li>" for r in result["reasons"]) or "<li>No issues found.</li>"
        details_html = " &nbsp;|&nbsp; ".join(
            f"<b>{k.replace('_',' ').title()}</b>: {v}"
            for k, v in result["details"].items() if k != "note"
        )
        st.markdown(f"""
        <div style="border:2px solid {border};border-radius:10px;padding:1rem;
                    background:{color};margin-bottom:1rem;">
            <h4 style="margin:0 0 0.3rem 0;">{icon} {title} &mdash; <span style="color:{border}">{status}</span></h4>
            <p style="margin:0 0 0.5rem 0;color:#444;font-size:0.9rem;">{description}</p>
            <ul style="margin:0 0 0.5rem 0;">{reasons_html}</ul>
            <p style="margin:0;font-size:0.8rem;color:#555;">{details_html}</p>
        </div>
        """, unsafe_allow_html=True)

    bias_card("🔄 Overtrading",     ot, "Trading too frequently — bursts within single hours, high volume vs balance, or rapid position flipping.")
    bias_card("😰 Loss Aversion",   la, "Holding losing trades too long while cutting winners short.")
    bias_card("😤 Revenge Trading", rt, "Opening oversized positions shortly after a loss to 'win back' money.")

    if rt["flagged_trades"]:
        st.markdown("**Flagged Revenge Trades:**")
        st.dataframe(pd.DataFrame(rt["flagged_trades"]), use_container_width=True)
        
         # ← ADD THIS BLOCK HERE
    bias_summary = f"""
    Trading Analysis Summary:
    - Total Trades: {len(df)}
    - Loss/Win Ratio: {loss_ratio:.2f}x
    - Peak Trading Hour: {peak_hour.strftime("%H:%M")}
    - Average Win: ${avg_win:.2f}, Average Loss: ${avg_loss:.2f}

    Overtrading: {"DETECTED" if ot["flagged"] else "CLEAR"}
    Reasons: {"; ".join(ot["reasons"]) if ot["reasons"] else "None"}

    Loss Aversion: {"DETECTED" if la["flagged"] else "CLEAR"}
    Reasons: {"; ".join(la["reasons"]) if la["reasons"] else "None"}

    Revenge Trading: {"DETECTED" if rt["flagged"] else "CLEAR"}
    Reasons: {"; ".join(rt["reasons"]) if rt["reasons"] else "None"}
    Revenge trade count: {rt["details"].get("revenge_trade_count", 0)}
    """

    # ── Gemini call helper (retry + model fallback) ──────────
    def gemini_call(contents, model, max_retries=3):
        FALLBACK = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash-8b"]
        to_try   = [model] + [m for m in FALLBACK if m != model]
        for current in to_try:
            for attempt in range(max_retries):
                try:
                    resp = client.models.generate_content(model=current, contents=contents)
                    if current != model:
                        st.caption(f"ℹ️ Used `{current}` (fallback — `{model}` quota exhausted).")
                    return resp.text
                except genai_errors.ClientError as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        m    = re.search(r"retry.*?(\d+)s", err, re.IGNORECASE)
                        wait = min(int(m.group(1)) if m else 2 ** attempt * 5, 60)
                        if attempt < max_retries - 1:
                            st.toast(f"⏳ Quota hit on `{current}`, retrying in {wait}s…")
                            time.sleep(wait)
                        else:
                            break
                    else:
                        raise
        return ("⚠️ All Gemini models hit their free-tier quota.\n"
                "Wait ~1 min and retry, or add billing at https://ai.google.dev/gemini-api/docs/rate-limits")

# ─────────────────────────────────────────────────────────────
# 9. AI CHATBOT COACH (CLEAN VERSION)
# ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 AI Trading Coach")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your biases..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response using our new file
        with st.spinner("Coach is thinking..."):
            full_response = get_chatbot_response(bias_summary, prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# ═════════════════════════════════════════════════════════════
# PAGE: LEARNING CENTRE
# ═════════════════════════════════════════════════════════════
elif page == "📚 Learning Centre":

    # ── Global styles (pure CSS, no Streamlit widgets inside HTML) ──
    st.markdown("""
    <style>
    /* Section header banners — used only for static HTML, no widgets inside */
    .section-banner {
        padding: 1.4rem 2rem;
        border-radius: 14px 14px 0 0;
        margin-bottom: 0;
    }
    .banner-ot { background: linear-gradient(90deg, #f9a825 0%, #ffcc02 100%); }
    .banner-la { background: linear-gradient(90deg, #c62828 0%, #e57373 100%); }
    .banner-rt { background: linear-gradient(90deg, #283593 0%, #5c6bc0 100%); }
    .banner-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    .banner-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }
    /* Info boxes — pure HTML only, no widgets inside */
    .info-quote {
        background: #f8f9fa;
        border-left: 5px solid #adb5bd;
        padding: 1rem 1.4rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        font-size: 1rem;
        color: #495057;
        margin: 1rem 0;
    }
    .info-tip {
        background: #d4edda;
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        color: #155724;
        font-size: 0.95rem;
        margin: 1rem 0;
    }
    .info-warn {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        color: #856404;
        font-size: 0.95rem;
        margin: 1rem 0;
    }
    /* Quiz styles */
    .quiz-banner {
        background: linear-gradient(90deg, #1a237e, #3949ab);
        border-radius: 14px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .quiz-banner h2 { color: white; font-size: 2rem; margin: 0 0 0.4rem; }
    .quiz-banner p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1rem; }
    .score-box {
        text-align: center;
        padding: 2rem;
        border-radius: 14px;
        margin: 1rem 0;
    }
    .score-great { background: #d4edda; border: 2px solid #28a745; }
    .score-ok    { background: #fff3cd; border: 2px solid #ffc107; }
    .score-low   { background: #f8d7da; border: 2px solid #dc3545; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📚 Learning Centre")
    st.markdown(
        "Understand the three psychological biases the NBC Bias Detector monitors — "
        "what they are, why they happen, what they cost, and exactly how the code spots them."
    )

    # ── Top-level section selector ──────────────────────────
    section = st.radio(
        "Jump to section:",
        ["🔄 Overtrading", "😰 Loss Aversion", "😤 Revenge Trading", "📋 Quick Reference", "🧠 Quiz"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # ════════════════════════════════════════════════════════
    # SECTION: OVERTRADING
    # ════════════════════════════════════════════════════════
    if section == "🔄 Overtrading":

        st.markdown("""
        <div class="section-banner banner-ot">
            <p class="banner-title">🔄 Overtrading</p>
            <p class="banner-subtitle">Bias 01 — Trading too frequently or with too much size relative to your account</p>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4, t5 = st.tabs([
            "📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"
        ])

        with t1:
            st.markdown("### What is Overtrading?")
            st.markdown(
                "Overtrading means executing **far more trades than a sound strategy justifies**. "
                "It comes in three flavours that often overlap:"
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.container(border=True):
                    st.markdown("#### ⏱️ Frequency")
                    st.markdown("Too many trades crammed into a single hour — firing at noise instead of signal.")
            with col2:
                with st.container(border=True):
                    st.markdown("#### 📦 Size")
                    st.markdown("Total notional volume wildly exceeds account size — overexposed to every market move.")
            with col3:
                with st.container(border=True):
                    st.markdown("#### 🔀 Position Flipping")
                    st.markdown("Rapidly alternating Buy → Sell → Buy on the same asset — chasing the price both ways.")
            st.markdown("""
            <div class="info-quote">
            "A disciplined trader fires only when their edge is clearly present. Overtraders fire any time they <em>feel</em> like something might happen."
            </div>
            """, unsafe_allow_html=True)

        with t2:
            st.markdown("### Why Does Overtrading Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**⚡ Action Bias**")
                    st.markdown("Humans are wired to *do something* when anxious. Sitting in cash feels like losing even when it's the correct position.")
                with st.container(border=True):
                    st.markdown("**😰 FOMO**")
                    st.markdown("Every price wiggle looks like a missed opportunity. The brain exaggerates the cost of inaction.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🎰 Dopamine Loops**")
                    st.markdown("Placing orders triggers the same reward circuits as gambling. The act of trading feels good independently of the outcome.")
                with st.container(border=True):
                    st.markdown("**😴 Boredom**")
                    st.markdown("Slow markets cause traders to manufacture setups that simply aren't there.")
            st.markdown("""
            <div class="info-quote">
            "The market is a device for transferring money from the impatient to the patient." — Warren Buffett
            </div>
            """, unsafe_allow_html=True)

        with t3:
            st.markdown("### What Does Overtrading Cost?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**Direct Costs**")
                    st.markdown(
                        "- Spreads and commissions compound with every unnecessary trade\n"
                        "- Slippage increases when entries are impulsive (bad fills)\n"
                        "- Costs accumulate whether the trade wins or loses"
                    )
            with col2:
                with st.container(border=True):
                    st.markdown("**Indirect Costs**")
                    st.markdown(
                        "- Cognitive fatigue degrades decision quality mid-session\n"
                        "- Overexposure — many open positions means one market move wipes several at once\n"
                        "- Drift from strategy — impulsive trades break your tested edge"
                    )
            st.markdown("""
            <div class="info-warn">
            ⚠️ <b>Real cost example:</b> A trader making 20 trades/day at $5 commission = $100/day in fees alone — 
            that's <b>$26,000/year</b> before a single market move is even considered.
            </div>
            """, unsafe_allow_html=True)

        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**Maya's morning session** — Account: $10,000")
                events = [
                    ("9:30 AM", "Buys 0.5 BTC", "Entry"),
                    ("9:38 AM", "Price dips — panic sells, immediately re-buys", "Switch"),
                    ("9:45 AM", "Buys ETH — price flat — sells 7 min later", "Churn"),
                    ("9:52 AM", "Re-enters BTC — exits 6 min later", "Churn"),
                    ("10:00 AM", "8 trades completed in 30 minutes", "Result"),
                ]
                for time, action, tag in events:
                    col1, col2, col3 = st.columns([1, 4, 1])
                    col1.markdown(f"`{time}`")
                    col2.markdown(action)
                    color = "🔴" if tag in ("Switch", "Churn") else ("🏁" if tag == "Result" else "⚪")
                    col3.markdown(color)
            st.markdown("""
            <div class="info-warn">
            Maya's account is down <b>1.2%</b> — not because the market moved against her, 
            but purely from transaction costs and bad fills on impulsive entries.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-tip">
            ✅ <b>The Fix:</b> Set a hard daily trade limit (e.g. max 5 trades/day). 
            Only enter when every item on your pre-trade checklist is met — not just because it "feels right."
            </div>
            """, unsafe_allow_html=True)

        with t5:
            st.markdown("### How `bias_engine.py` Detects Overtrading")
            st.info("Three independent sub-checks run in parallel — **any one** can flag the bias.", icon="ℹ️")

            with st.expander("📊 Sub-check A — Trades per hour  (`timestamp`)", expanded=True):
                st.markdown(
                    "We resample the trade log into **1-hour buckets** using `pd.Grouper`. "
                    "The threshold is **auto-calculated as 2× your own average hourly rate** "
                    "so it's always personalised — a high-frequency trader gets a higher threshold than a swing trader."
                )
                st.code("""
hourly   = df.set_index("timestamp").resample("1h").size()
peak_val = int(hourly.max())

# Threshold auto-set to 2× the trader's avg hourly rate (min 15)
total_hours  = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
avg_hourly   = len(df) / total_hours
max_per_hour = max(int(avg_hourly * 2), 15)

if peak_val > max_per_hour:
    result["flagged"] = True
    result["reasons"].append(f"Executed {peak_val} trades in one hour (threshold: {max_per_hour})")
                """, language="python")
                st.caption("Columns used: `timestamp`")

            with st.expander("📊 Sub-check B — Volume / balance ratio  (`quantity`, `entry_price`, `balance`)"):
                st.markdown(
                    "We calculate total notional traded (`quantity × entry_price`) and divide by average account balance. "
                    "A ratio above 3× means the trader is financially overexposed to market risk."
                )
                st.code("""
df["trade_value"] = df["quantity"] * df["entry_price"]
total_vol   = df["trade_value"].sum()
avg_balance = df["balance"].mean()
ratio       = total_vol / avg_balance

if ratio > max_vol_ratio:   # default threshold: 3.0×
    result["flagged"] = True
    result["reasons"].append(f"Total volume is {ratio:.1f}× your average balance")
                """, language="python")
                st.caption("Columns used: `quantity`, `entry_price`, `balance`")

            with st.expander("📊 Sub-check C — Rapid position switching  (`asset`, `buy_sell`, `timestamp`)"):
                st.markdown(
                    "For each asset, we scan consecutive trades. If the direction flipped "
                    "(Buy→Sell or Sell→Buy) within 30 minutes, it counts as a switch. "
                    "Three or more switches = position-flip overtrading flagged."
                )
                st.code("""
switches = 0
for asset in df["asset"].unique():
    sub = df[df["asset"] == asset].reset_index(drop=True)
    for i in range(1, len(sub)):
        diff_min = (sub.loc[i, "timestamp"]
                    - sub.loc[i-1, "timestamp"]).total_seconds() / 60
        if sub.loc[i, "buy_sell"] != sub.loc[i-1, "buy_sell"] and diff_min < 30:
            switches += 1

if switches >= 3:   # default threshold: 3 switches
    result["flagged"] = True
    result["reasons"].append(f"Detected {switches} rapid position switches")
                """, language="python")
                st.caption("Columns used: `timestamp`, `asset`, `buy_sell`")

    # ════════════════════════════════════════════════════════
    # SECTION: LOSS AVERSION
    # ════════════════════════════════════════════════════════
    elif section == "😰 Loss Aversion":

        st.markdown("""
        <div class="section-banner banner-la">
            <p class="banner-title">😰 Loss Aversion</p>
            <p class="banner-subtitle">Bias 02 — Holding losing trades too long while cutting winners short</p>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4, t5 = st.tabs([
            "📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"
        ])

        with t1:
            st.markdown("### What is Loss Aversion?")
            st.markdown(
                "Loss aversion is the cognitive bias where the **pain of losing feels roughly twice as powerful** "
                "as the pleasure of an equivalent gain *(Kahneman & Tversky, 1979)*. "
                "In trading it produces two opposite but equally damaging mistakes:"
            )
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("#### ✂️ Cutting Winners Short")
                    st.markdown(
                        "Closing profitable trades too early out of fear the profit will disappear. "
                        "Result: **small average wins**."
                    )
            with col2:
                with st.container(border=True):
                    st.markdown("#### 🪝 Riding Losers Long")
                    st.markdown(
                        "Refusing to close a losing trade while hoping it will recover. "
                        "Result: **large average losses**."
                    )
            st.markdown("""
            <div class="info-quote">
            "An open losing trade isn't officially a loss yet. Closing it makes it real — 
            and loss aversion makes that reality feel unbearable."
            </div>
            """, unsafe_allow_html=True)

        with t2:
            st.markdown("### Why Does Loss Aversion Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**🛡️ Ego Protection**")
                    st.markdown("An open losing trade isn't officially a loss yet. Closing it makes it real and forces you to confront the mistake.")
                with st.container(border=True):
                    st.markdown("**⚓ Anchoring**")
                    st.markdown("Traders anchor to their entry price and wait for 'break even' — even as the loss compounds further below.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🕳️ Sunk-Cost Fallacy**")
                    st.markdown('"I\'ve already lost $500 — I can\'t close now." The money already lost influences decisions about the future, irrationally.')
                with st.container(border=True):
                    st.markdown("**📉 Prospect Theory**")
                    st.markdown("Our brains weigh losses 2× more than gains. A $100 loss hurts as much as a $200 gain feels good — causing asymmetric behaviour.")
            st.markdown("""
            <div class="info-quote">
            "The first loss is the best loss." — Trading floor adage meaning cutting losses quickly 
            is almost always the mathematically superior decision.
            </div>
            """, unsafe_allow_html=True)

        with t3:
            st.markdown("### The Mathematical Damage")
            st.markdown("With **avg win = $80** and **avg loss = $200** (2.5× ratio), here's what happens at different win rates:")
            data = {
                "Win Rate": ["50%", "60%", "70%", "80%"],
                "Expected Value Per Trade": ["−$60.00", "−$32.00", "−$4.00", "+$24.00"],
                "Result After 100 Trades": ["−$6,000", "−$3,200", "−$400", "+$2,400"],
                "Verdict": ["🔴 Losing", "🔴 Losing", "🔴 Losing", "🟢 Profitable"],
            }
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.markdown("""
            <div class="info-warn">
            ⚠️ Most retail traders have a win rate below 60%. With a 2.5× loss ratio they are 
            <b>mathematically guaranteed to lose money in the long run</b> — regardless of how 
            often they are "right" about market direction.
            </div>
            """, unsafe_allow_html=True)

        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**James's trading day** — Asset: ETH")
                events = [
                    ("10:00", "Buys ETH at $3,000", "neutral"),
                    ("10:45", "ETH rises to $3,120 (+4%). James fears a reversal — exits. Profit: +$120", "win"),
                    ("11:00", "ETH continues to $3,300. James re-enters at $3,280", "neutral"),
                    ("13:30", "ETH drops to $2,950. James thinks 'it'll bounce' — holds", "loss"),
                    ("16:00", "ETH drops to $2,700. James finally closes. Loss: −$580", "loss"),
                ]
                for time, action, tag in events:
                    col1, col2 = st.columns([1, 5])
                    col1.markdown(f"`{time}`")
                    icon = "🟢" if tag == "win" else ("🔴" if tag == "loss" else "⚪")
                    col2.markdown(f"{icon} {action}")
            st.markdown("""
            <div class="info-warn">
            James had a <b>60% win rate</b> that day — but his average win was $120 and average loss was $580. 
            Net result: deeply negative despite being right more often than wrong.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-tip">
            ✅ <b>The Fix:</b> Set a hard stop-loss AND a profit target before every trade. 
            Use a minimum 1:1.5 risk/reward ratio. <b>Never move your stop-loss further away after entry.</b>
            </div>
            """, unsafe_allow_html=True)

        with t5:
            st.markdown("### How `bias_engine.py` Detects Loss Aversion")
            st.info("Two sub-checks — both look for the same pattern from different angles.", icon="ℹ️")

            with st.expander("📊 Sub-check A — Win/Loss P/L size ratio  (`profit_loss`)", expanded=True):
                st.markdown(
                    "We separate all trades into **winners** (`profit_loss > 0`) and **losers** (`profit_loss < 0`), "
                    "compute the mean of each group, and compare. A ratio above 1.5× is flagged."
                )
                st.code("""
winners  = df[df["profit_loss"] > 0]["profit_loss"]
losers   = df[df["profit_loss"] < 0]["profit_loss"].abs()

avg_win  = winners.mean()
avg_loss = losers.mean()
ratio    = avg_loss / avg_win   # e.g. 2.1 = avg loss is 2.1× avg win

if ratio > ratio_threshold:     # default threshold: 1.5×
    result["flagged"] = True
    result["reasons"].append(
        f"Average loss (${avg_loss:.2f}) is {ratio:.1f}× your average win (${avg_win:.2f})"
    )
                """, language="python")
                st.caption("Columns used: `profit_loss`")

            with st.expander("📊 Sub-check B — Price range asymmetry  (`entry_price`, `exit_price`, `profit_loss`)"):
                st.markdown(
                    "Even if P/L isn't heavily skewed yet, we check how far price *physically travelled* "
                    "before the trader exited. If losers travel 1.5× more distance than winners before exit, "
                    "it's a strong signal of holding losses and cutting profits."
                )
                st.code("""
df["price_range"] = (df["exit_price"] - df["entry_price"]).abs()

avg_range_win  = df[df["profit_loss"] > 0]["price_range"].mean()
avg_range_loss = df[df["profit_loss"] < 0]["price_range"].mean()
range_ratio    = avg_range_loss / avg_range_win

if range_ratio > price_range_ratio:   # default threshold: 1.5×
    result["flagged"] = True
    result["reasons"].append(
        f"Losing trades travel {range_ratio:.1f}× more price distance before exit"
    )
                """, language="python")
                st.caption("Columns used: `entry_price`, `exit_price`, `profit_loss`")

    # ════════════════════════════════════════════════════════
    # SECTION: REVENGE TRADING
    # ════════════════════════════════════════════════════════
    elif section == "😤 Revenge Trading":

        st.markdown("""
        <div class="section-banner banner-rt">
            <p class="banner-title">😤 Revenge Trading</p>
            <p class="banner-subtitle">Bias 03 — Opening oversized positions right after a loss to "win back" money</p>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3, t4, t5 = st.tabs([
            "📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"
        ])

        with t1:
            st.markdown("### What is Revenge Trading?")
            st.markdown(
                "Revenge trading is the impulsive act of placing a **larger-than-normal trade "
                "immediately after a loss**, driven by the emotional desire to recover the money quickly. "
                "It has two defining hallmarks:"
            )
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("#### ⚡ Timing")
                    st.markdown("The new trade opens **very shortly** after the losing trade closes — often within minutes, before any rational analysis can take place.")
            with col2:
                with st.container(border=True):
                    st.markdown("#### 📏 Size")
                    st.markdown("The new position is **significantly larger** than the trader's historical average — magnifying the risk at exactly the worst emotional moment.")
            st.markdown("""
            <div class="info-quote">
            "The market does not know or care that you just lost money — and it will not give it back on demand."
            </div>
            """, unsafe_allow_html=True)

        with t2:
            st.markdown("### Why Does Revenge Trading Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**🎲 Gambler's Fallacy**")
                    st.markdown('"I just lost, so I\'m *due* a win." The market has no memory — previous outcomes do not change future probabilities.')
                with st.container(border=True):
                    st.markdown("**🪞 Ego Threat**")
                    st.markdown("A loss feels like personal failure. A quick, large recovery trade would restore self-image and prove the original instinct right.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🧠 Emotional Hijacking**")
                    st.markdown("The amygdala (fight-or-flight) fires after a loss and overrides the prefrontal cortex (rational planning) — decisions become reactive, not analytical.")
                with st.container(border=True):
                    st.markdown("**📌 Recency Bias**")
                    st.markdown("The last trade feels more significant than the statistical average. One loss dominates thinking and distorts risk assessment.")
            st.markdown("""
            <div class="info-quote">
            "After a loss, the worst thing you can do is try to make it back immediately. 
            The market punishes impatience twice." — Mark Douglas, <em>Trading in the Zone</em>
            </div>
            """, unsafe_allow_html=True)

        with t3:
            st.markdown("### The Compounding Damage")
            data = {
                "Trade": ["Normal loss", "Revenge trade (3× size)", "Total damage"],
                "Position Size": ["1% of account", "3% of account", "—"],
                "Outcome": ["Loss", "Loss", "—"],
                "P/L": ["−$100", "−$300", "−$400"],
                "vs. Stopping After Loss": ["−$100", "—", "4× worse"],
            }
            st.dataframe(data, use_container_width=True, hide_index=True)
            st.markdown("""
            <div class="info-warn">
            ⚠️ Revenge trades are placed <b>without proper analysis</b> — so their win rate is 
            <em>lower</em> than normal trades. You take bigger size on worse setups at your most 
            emotionally compromised moment.
            </div>
            """, unsafe_allow_html=True)

        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**Sofia's afternoon** — Normal position size: 0.3 BTC")
                events = [
                    ("2:10 PM", "Buys 0.3 BTC at $42,000. Price falls.", "neutral"),
                    ("2:12 PM", "Exits at $41,500. Loss: −$150", "loss"),
                    ("2:14 PM", "Angry — buys 0.9 BTC (3× normal size) at $41,500 to 'get it back'", "revenge"),
                    ("2:28 PM", "Price falls to $40,800. Exits. Loss: −$630", "loss"),
                    ("2:28 PM", "Total damage in 18 minutes: −$780 instead of −$150", "result"),
                ]
                for time, action, tag in events:
                    col1, col2 = st.columns([1, 5])
                    col1.markdown(f"`{time}`")
                    icon = "🔴" if tag == "loss" else ("🚨" if tag == "revenge" else ("💥" if tag == "result" else "⚪"))
                    col2.markdown(f"{icon} {action}")
            st.markdown("""
            <div class="info-warn">
            Sofia's revenge trade didn't recover the loss — it multiplied it by <b>5.2×</b>.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-tip">
            ✅ <b>The Fix:</b> Mandatory 30-minute cooling-off rule after any loss. Step away from the screen. 
            Only re-enter if a pre-defined setup is present on your checklist — not because you "need to win back" money.
            </div>
            """, unsafe_allow_html=True)

        with t5:
            st.markdown("### How `bias_engine.py` Detects Revenge Trading")
            st.info("Uses Pandas `.shift()` to compare each trade against the one immediately before it — no slow Python loops.", icon="ℹ️")

            with st.expander("📊 Full detection logic — step by step", expanded=True):
                st.markdown("**Step 1 — Sort chronologically and compute average quantity:**")
                st.code("""
df      = df.sort_values("timestamp").reset_index(drop=True)
avg_qty = df["quantity"].mean()   # the trader's historical normal trade size
                """, language="python")

                st.markdown("**Step 2 — `.shift(1)` pulls the prior trade's data into each row (vectorised — no loop):**")
                st.code("""
df["prev_pl"]         = df["profit_loss"].shift(1)   # prior trade's P/L result
df["prev_ts"]         = df["timestamp"].shift(1)      # prior trade's timestamp
df["time_since_prev"] = (
    df["timestamp"] - df["prev_ts"]
).dt.total_seconds() / 60                            # gap in minutes
                """, language="python")

                st.markdown("**Step 3 — Three-condition mask flags revenge trades in a single vectorised pass:**")
                st.code("""
mask = (
    (df["prev_pl"] < 0)                              # prior trade was a loss
    & (df["quantity"] > avg_qty * qty_multiplier)    # this trade is oversized (default: 1.5×)
    & (df["time_since_prev"] <= time_window_min)     # opened within time window (default: 15 min)
)
flagged_df = df[mask]   # every matching row is a confirmed revenge trade
                """, language="python")
                st.caption("Columns used: `profit_loss`, `quantity`, `timestamp`")

            with st.expander("💡 Why .shift() instead of a for-loop?"):
                st.markdown(
                    "`.shift(1)` moves the entire column down by one row in a **single vectorised operation** — "
                    "no Python `for` loop, no row-by-row iteration. It runs instantly even on tens of thousands of trades."
                )
                st.code("""
# What .shift(1) does to your data:

# Before:                    After shift(1):
# row 0: profit_loss = +50   row 0: prev_pl = NaN    ← no prior trade
# row 1: profit_loss = -120  row 1: prev_pl = +50
# row 2: profit_loss = +80   row 2: prev_pl = -120   ← loss before → check size!
# row 3: profit_loss = -200  row 3: prev_pl = +80
                """, language="python")

    # ════════════════════════════════════════════════════════
    # SECTION: QUICK REFERENCE
    # ════════════════════════════════════════════════════════
    elif section == "📋 Quick Reference":

        st.markdown("## 📋 Quick Reference — All Three Biases")
        st.markdown("A single-page cheat sheet comparing all three biases side-by-side.")
        st.divider()

        col1, col2, col3 = st.columns(3)
        cards = [
            ("col1", "🔄 Overtrading", "#f9a825",
             "Too many / too large trades",
             "FOMO, boredom, action bias, dopamine loops",
             "`timestamp` `quantity` `balance` `buy_sell` `asset`",
             "`resample('1h')`, volume ratio, position-switch loop",
             "2× your avg hourly rate, or volume >3× balance",
             "Hard daily trade limit + pre-trade checklist"),
            ("col2", "😰 Loss Aversion", "#c62828",
             "Hold losers, cut winners",
             "Ego protection, anchoring, sunk-cost fallacy",
             "`profit_loss` `entry_price` `exit_price`",
             "`mean()` on filtered slices, price range comparison",
             "Loss/win P/L ratio >1.5×",
             "Pre-set stop-loss + take-profit before every trade"),
            ("col3", "😤 Revenge Trading", "#283593",
             "Oversized trade right after a loss",
             "Gambler's fallacy, ego threat, emotional hijacking",
             "`profit_loss` `quantity` `timestamp`",
             "`.shift(1)` vectorised three-condition mask",
             "Qty >1.5× avg within 15 min of a loss",
             "Mandatory 30-min cooling-off after any loss"),
        ]
        for col_name, title, color, behaviour, driver, fields, technique, threshold, fix in cards:
            col = locals()[col_name]
            with col:
                with st.container(border=True):
                    st.markdown(f"<h3 style='color:{color};margin-bottom:0.8rem'>{title}</h3>", unsafe_allow_html=True)
                    st.markdown("**Core behaviour**")
                    st.markdown(behaviour)
                    st.markdown("**Emotional driver**")
                    st.markdown(driver)
                    st.markdown("**CSV fields used**")
                    st.markdown(fields)
                    st.markdown("**Pandas technique**")
                    st.markdown(technique)
                    st.markdown("**Detection threshold**")
                    st.markdown(threshold)
                    st.markdown(f"""<div class="info-tip">✅ <b>Quick fix:</b> {fix}</div>""", unsafe_allow_html=True)

        st.divider()
        st.caption("NBC Bias Detector · Learning Centre · Educational only, not financial advice.")

    # ════════════════════════════════════════════════════════
    # SECTION: QUIZ
    # ════════════════════════════════════════════════════════
    elif section == "🧠 Quiz":

        st.markdown("""
        <div class="quiz-banner">
            <h2>🧠 Knowledge Quiz</h2>
            <p>Test your understanding of the three trading biases · 10 questions · Instant feedback</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Quiz questions ───────────────────────────────────
        questions = [
            {
                "q": "A trader executes 18 trades in a single hour. Which bias does this most directly indicate?",
                "options": ["Loss Aversion", "Revenge Trading", "Overtrading", "Anchoring Bias"],
                "answer": "Overtrading",
                "explanation": "Time-based clustering — too many trades in a single hour — is the primary signal of **Overtrading**. Our engine flags this using `resample('1h')`.",
            },
            {
                "q": "Sarah's average winning trade returns $90, but her average losing trade costs her $220. What is her loss/win ratio and what does it indicate?",
                "options": [
                    "0.41× — she is managing risk well",
                    "2.44× — she likely has Loss Aversion",
                    "1.0× — perfectly balanced",
                    "2.44× — she likely has Revenge Trading tendencies",
                ],
                "answer": "2.44× — she likely has Loss Aversion",
                "explanation": "Loss/win ratio = 220 ÷ 90 = **2.44×**. Our threshold is 1.5×. A ratio above 1.5× indicates **Loss Aversion** — holding losers too long while cutting winners short.",
            },
            {
                "q": "In `bias_engine.py`, which Pandas method is used to compare each trade against the immediately preceding one without using a for-loop?",
                "options": ["`df.merge()`", "`df.shift(1)`", "`df.rolling(1)`", "`df.diff(1)`"],
                "answer": "`df.shift(1)`",
                "explanation": "`.shift(1)` moves the entire column down by one row in a single vectorised operation — placing the prior trade's data in the current row for comparison.",
            },
            {
                "q": "Tom loses $200 on a trade at 2:05 PM. At 2:09 PM he opens a new position at 3× his normal size. What bias is this?",
                "options": ["Overtrading", "Loss Aversion", "Confirmation Bias", "Revenge Trading"],
                "answer": "Revenge Trading",
                "explanation": "A significantly oversized trade opened within minutes of a loss is the textbook definition of **Revenge Trading**. Our engine uses a `.shift(1)` mask to catch exactly this pattern.",
            },
            {
                "q": "Which psychological theory explains why a $100 loss feels roughly as painful as a $200 gain feels good?",
                "options": [
                    "Efficient Market Hypothesis",
                    "Prospect Theory (Kahneman & Tversky)",
                    "Random Walk Theory",
                    "Modern Portfolio Theory",
                ],
                "answer": "Prospect Theory (Kahneman & Tversky)",
                "explanation": "**Prospect Theory** (1979) established that losses loom approximately 2× larger than equivalent gains in human subjective experience — the core mechanism behind Loss Aversion.",
            },
            {
                "q": "The NBC Bias Detector auto-calculates the overtrading threshold as 2× your average hourly trade rate. If you average 4 trades/hour, what threshold is set?",
                "options": ["4", "6", "8", "15"],
                "answer": "8",
                "explanation": "2 × 4 = **8** trades/hour. However, there is a minimum of 15 — so if 2× your average is below 15, the threshold is capped at 15 to avoid false positives for very low-frequency traders.",
            },
            {
                "q": "A trader rapidly buys and sells the same asset three times in 20 minutes, alternating direction each time. Which overtrading sub-check catches this?",
                "options": [
                    "Sub-check A — Trades per hour",
                    "Sub-check B — Volume / balance ratio",
                    "Sub-check C — Rapid position switching",
                    "None — this is not overtrading",
                ],
                "answer": "Sub-check C — Rapid position switching",
                "explanation": "**Sub-check C** specifically looks for Buy→Sell→Buy flips on the same asset within a 30-minute window. Three or more such flips triggers the flag.",
            },
            {
                "q": "Emma has a 65% win rate but is still losing money overall. What is the most likely explanation?",
                "options": [
                    "Her win rate calculation is wrong",
                    "She is experiencing Overtrading — too many commission costs",
                    "Her average loss is significantly larger than her average win (Loss Aversion)",
                    "She is trading the wrong assets",
                ],
                "answer": "Her average loss is significantly larger than her average win (Loss Aversion)",
                "explanation": "A 65% win rate with avg win $80 and avg loss $200 = `0.65×80 − 0.35×200 = −$18/trade`. **Loss Aversion** creates a negative expected value even at above-average win rates.",
            },
            {
                "q": "Which column in the CSV does the revenge trading detector use to measure the gap between a loss and the next trade?",
                "options": ["`profit_loss`", "`entry_price`", "`timestamp`", "`balance`"],
                "answer": "`timestamp`",
                "explanation": "After using `.shift(1)` to bring the prior trade's timestamp into the current row, the engine calculates `(current_timestamp - prev_timestamp).dt.total_seconds() / 60` to get the gap in minutes.",
            },
            {
                "q": "Which of these is NOT one of the three biases detected by the NBC Bias Detector?",
                "options": ["Overtrading", "Confirmation Bias", "Loss Aversion", "Revenge Trading"],
                "answer": "Confirmation Bias",
                "explanation": "The NBC Bias Detector specifically monitors **Overtrading**, **Loss Aversion**, and **Revenge Trading**. Confirmation Bias (seeking information that confirms existing beliefs) is a real trading bias but is not detected by this tool.",
            },
        ]

        # Initialise session state for quiz
        if "quiz_answers" not in st.session_state:
            st.session_state.quiz_answers = {}
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False

        if not st.session_state.quiz_submitted:
            # ── Render questions ─────────────────────────────
            for i, q in enumerate(questions):
                with st.container(border=True):
                    st.markdown(f"**Question {i+1} of {len(questions)}**")
                    st.markdown(f"#### {q['q']}")
                    answer = st.radio(
                        f"q{i}",
                        q["options"],
                        index=None,
                        label_visibility="collapsed",
                        key=f"quiz_q{i}",
                    )
                    if answer:
                        st.session_state.quiz_answers[i] = answer

            st.markdown("")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                answered = len(st.session_state.quiz_answers)
                st.markdown(f"*{answered} of {len(questions)} questions answered*")
                if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                    if answered < len(questions):
                        st.warning("Please answer all questions before submitting.")
                    else:
                        st.session_state.quiz_submitted = True
                        st.rerun()
        else:
            # ── Show results ─────────────────────────────────
            score = sum(
                1 for i, q in enumerate(questions)
                if st.session_state.quiz_answers.get(i) == q["answer"]
            )
            pct = score / len(questions) * 100

            if pct >= 80:
                box_class = "score-great"
                emoji = "🏆"
                verdict = "Excellent! You have a strong grasp of trading psychology."
            elif pct >= 60:
                box_class = "score-ok"
                emoji = "📈"
                verdict = "Good effort! Review the sections where you made mistakes."
            else:
                box_class = "score-low"
                emoji = "📚"
                verdict = "Keep studying! Head back through the Learning Centre sections."

            st.markdown(f"""
            <div class="score-box {box_class}">
                <h1>{emoji}</h1>
                <h2>{score} / {len(questions)} correct ({pct:.0f}%)</h2>
                <p>{verdict}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-question feedback ─────────────────────────
            st.markdown("### Detailed Feedback")
            for i, q in enumerate(questions):
                user_ans = st.session_state.quiz_answers.get(i, "Not answered")
                correct  = user_ans == q["answer"]
                with st.container(border=True):
                    icon = "✅" if correct else "❌"
                    st.markdown(f"{icon} **Q{i+1}: {q['q']}**")
                    if not correct:
                        st.markdown(f"Your answer: ~~{user_ans}~~")
                    st.markdown(f"**Correct answer: {q['answer']}**")
                    st.info(q["explanation"], icon="💡")

            st.markdown("")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Retake Quiz", use_container_width=True):
                    st.session_state.quiz_answers  = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()