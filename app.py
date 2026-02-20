from ai_coach import get_chatbot_response
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import re
from datetime import datetime
from google import genai
from google.genai import errors as genai_errors
from bias_engine import run_all

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBC Bias Detector", layout="wide")

# ─────────────────────────────────────────────────────────────
# API SETUP
# ─────────────────────────────────────────────────────────────
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client  = genai.Client(api_key=API_KEY)
except Exception:
    st.error("Please set your GEMINI_API_KEY in Streamlit Secrets.")
    client = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 NBC Bias Detector")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "📋 Feedback & Recommendations", "📚 Learning Centre"],
        label_visibility="collapsed",
    )

    st.divider()

    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Upload all Trading CSVs",
        type="csv",
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2. Detection Thresholds")
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


# ─────────────────────────────────────────────────────────────
# LOAD & CACHE DATA (runs regardless of page)
# ─────────────────────────────────────────────────────────────
if uploaded_files:
    file_fingerprint = [(f.name, f.size) for f in uploaded_files]
    if st.session_state.get("file_fingerprint") != file_fingerprint:
        dfs    = [pd.read_csv(f) for f in uploaded_files]
        df_raw = pd.concat(dfs, ignore_index=True)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
        st.session_state["df"]               = df_raw
        st.session_state["file_fingerprint"] = file_fingerprint

# ─────────────────────────────────────────────────────────────
# SHARED: compute max_per_hour (used by Dashboard + Feedback)
# ─────────────────────────────────────────────────────────────
if "df" in st.session_state:
    df = st.session_state["df"]
    total_hours_global = max(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600, 1
    )
    avg_hourly_volume_global = len(df) / total_hours_global
    max_per_hour = st.sidebar.number_input(
        "Max trades/hour (Overtrading)",
        min_value=1,
        value=max(int(avg_hourly_volume_global * 2), 15),
        help=f"Auto-set to 2× your avg hourly rate ({avg_hourly_volume_global:.1f} trades/hr). Adjust as needed.",
    )
else:
    max_per_hour = 15  # safe default before data is loaded


# ─────────────────────────────────────────────────────────────
# GEMINI HELPER
# ─────────────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.title("🏦 National Bank: AI Bias Detector")

    if "df" not in st.session_state:
        st.info("⬅️ Upload one or more CSV files in the sidebar to begin analysis.")
        st.stop()

    df = st.session_state["df"]

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
            range_color=[0, int(max_per_hour)],
            color_continuous_scale="RdYlGn_r",
            range_y=[0, chart_ceiling],
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

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

        with st.spinner("Coach is thinking..."):
            full_response = get_chatbot_response(bias_summary, prompt)

        with st.chat_message("assistant"):
            st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


# ═════════════════════════════════════════════════════════════
# PAGE: FEEDBACK & RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════
elif page == "📋 Feedback & Recommendations":

    if "df" not in st.session_state:
        st.info("⬅️ Upload one or more CSV files in the sidebar to begin analysis.")
        st.stop()

    df = st.session_state["df"]

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

    # ── Pre-compute stats ────────────────────────────────────
    wins   = df[df["profit_loss"] > 0]["profit_loss"]
    losses = df[df["profit_loss"] < 0]["profit_loss"].abs()

    avg_win    = wins.mean()   if not wins.empty   else 0
    avg_loss   = losses.mean() if not losses.empty else 0
    loss_ratio = avg_loss / avg_win if avg_win > 0 else 0
    win_rate   = len(wins) / len(df) * 100 if len(df) > 0 else 0

    total_hours  = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600, 1)
    avg_hourly   = len(df) / total_hours
    revenge_count = rt["details"].get("revenge_trade_count", 0)

    hourly        = df.set_index("timestamp").resample("1h").size()
    peak_hour_val = int(hourly.max()) if not hourly.empty else 0

    # Severity scores 0-100
    ot_score      = min(100, (peak_hour_val / max(avg_hourly * 2, 1)) * 50) if ot["flagged"] else 10
    la_score      = min(100, (loss_ratio / 1.5) * 50)                        if la["flagged"] else 10
    rt_score      = min(100, revenge_count * 20)                              if rt["flagged"] else 5
    overall_score = (ot_score + la_score + rt_score) / 3

    def _severity(score):
        if score >= 75: return "🔴 Critical", "#c62828"
        if score >= 50: return "🟠 Moderate", "#e65100"
        if score >= 25: return "🟡 Mild",     "#f9a825"
        return "🟢 Clear", "#2e7d32"

    # ── Styles ───────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
    .fb-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 60%, #0d2137 100%);
        border-radius: 20px; padding: 2.5rem 2.5rem 2rem; margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.07); position: relative; overflow: hidden;
    }
    .fb-hero::before {
        content: ''; position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .fb-hero h1 { font-family: 'DM Serif Display', serif; color: #f0f4f8; font-size: 2rem; margin: 0 0 0.3rem; }
    .fb-hero p  { color: rgba(200,220,240,0.7); margin: 0; font-size: 0.95rem; }
    .gauge-card {
        border-radius: 16px; padding: 1.4rem; text-align: center;
        background: #111d28; border: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;
    }
    .gauge-title { font-size: 0.8rem; color: #90a4b7; text-transform: uppercase; letter-spacing: 0.08em; }
    .gauge-val   { font-family: 'DM Serif Display', serif; font-size: 2.6rem; margin: 0.2rem 0; }
    .gauge-badge { font-size: 0.85rem; font-weight: 600; }
    .rec-card {
        border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        border-left: 5px solid; background: #f8fafc;
    }
    .rec-card h4 { margin: 0 0 0.5rem; font-size: 1rem; }
    .rec-card p  { margin: 0; color: #444; font-size: 0.93rem; line-height: 1.55; }
    .insight-box {
        background: linear-gradient(135deg, #e8f4fd, #dbeafe); border: 1px solid #93c5fd;
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
        font-size: 0.93rem; color: #1e3a5f;
    }
    .insight-box strong { color: #1d4ed8; }
    .journal-card {
        background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px;
        padding: 1.2rem 1.4rem; margin-bottom: 0.8rem; font-size: 0.93rem;
        color: #78350f; line-height: 1.6;
    }
    .section-head {
        font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: #1a2a3a;
        margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 2px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fb-hero">
        <h1>📋 Feedback &amp; Recommendations</h1>
        <p>Personalised analysis of your trading psychology — actionable suggestions backed by your own data.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Bias Health Dashboard ────────────────────
    st.markdown('<p class="section-head">🩺 Bias Health Dashboard</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def gauge_card(col, title, score, label, color):
        col.markdown(f"""
        <div class="gauge-card">
            <div class="gauge-title">{title}</div>
            <div class="gauge-val" style="color:{color};">{score:.0f}</div>
            <div class="gauge-badge" style="color:{color};">{label}</div>
            <div style="margin-top:0.6rem;background:#1e2d3d;border-radius:99px;height:6px;overflow:hidden;">
                <div style="width:{min(score,100)}%;height:100%;background:{color};border-radius:99px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    overall_label, overall_color = _severity(overall_score)
    ot_label, ot_color           = _severity(ot_score)
    la_label, la_color           = _severity(la_score)
    rt_label, rt_color           = _severity(rt_score)

    gauge_card(col1, "Overall Risk Score", overall_score, overall_label, overall_color)
    gauge_card(col2, "🔄 Overtrading",     ot_score,      ot_label,      ot_color)
    gauge_card(col3, "😰 Loss Aversion",   la_score,      la_label,      la_color)
    gauge_card(col4, "😤 Revenge Trading", rt_score,      rt_label,      rt_color)

    fig_radar = go.Figure(go.Scatterpolar(
        r=[ot_score, la_score, rt_score, ot_score],
        theta=["Overtrading", "Loss Aversion", "Revenge Trading", "Overtrading"],
        fill="toself", fillcolor="rgba(220,38,38,0.15)",
        line=dict(color="#ef4444", width=2), name="Your Bias Score",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[30, 30, 30, 30],
        theta=["Overtrading", "Loss Aversion", "Revenge Trading", "Overtrading"],
        fill="toself", fillcolor="rgba(34,197,94,0.08)",
        line=dict(color="#22c55e", width=1.5, dash="dot"), name="Healthy Benchmark",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0f1923",
            radialaxis=dict(range=[0, 100], showticklabels=True,
                            tickfont=dict(color="#94a3b8"), gridcolor="#1e2d3d"),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=13), gridcolor="#1e2d3d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8")),
        margin=dict(l=60, r=60, t=30, b=30), height=340,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── SECTION 2: Bias Summaries ────────────────────────────
    st.markdown('<p class="section-head">🗣️ Your Bias Summaries</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🔄 Overtrading")
        if ot["flagged"]:
            peak_dt = hourly.idxmax()
            st.error(
                f"**You are overtrading.** Your peak hour hit **{peak_hour_val} trades** — "
                f"above your personalised threshold. You average **{avg_hourly:.1f} trades/hour**, "
                f"suggesting frequent entries without high-conviction setups. "
                f"Your highest-frequency hour was **{peak_dt.strftime('%A %d %b, %H:%M')}**.",
                icon="🚨",
            )
        else:
            st.success(
                f"**Overtrading: Clear.** Your average of {avg_hourly:.1f} trades/hour is within healthy bounds.",
                icon="✅",
            )

    with st.container(border=True):
        st.markdown("#### 😰 Loss Aversion")
        if la["flagged"]:
            ev_per_trade = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss
            st.error(
                f"**Loss aversion detected.** Your average loss (${avg_loss:.2f}) is "
                f"**{loss_ratio:.1f}× your average win** (${avg_win:.2f}). "
                f"With a {win_rate:.0f}% win rate, your expected value per trade is "
                f"**${ev_per_trade:+.2f}**. "
                f"{'This is negative — you are losing money in expectation despite winning more than half your trades.' if ev_per_trade < 0 else 'The gap between wins and losses creates long-term fragility.'}",
                icon="🚨",
            )
        else:
            st.success(
                f"**Loss Aversion: Clear.** Win/loss ratio is healthy ({loss_ratio:.2f}×).",
                icon="✅",
            )

    with st.container(border=True):
        st.markdown("#### 😤 Revenge Trading")
        if rt["flagged"]:
            st.error(
                f"**Revenge trading detected — {revenge_count} instance(s).** "
                f"After a loss, you opened an oversized position within the alert window. "
                f"These trades are placed at your most emotionally compromised moment.",
                icon="🚨",
            )
            if rt.get("flagged_trades"):
                st.dataframe(pd.DataFrame(rt["flagged_trades"]), use_container_width=True, hide_index=True)
        else:
            st.success("**Revenge Trading: Clear.** No oversized positions detected immediately after losses.", icon="✅")

    # ── SECTION 3: Graphical Insights ───────────────────────
    st.markdown('<p class="section-head">📊 Graphical Insights</p>', unsafe_allow_html=True)

    tab_heat, tab_time, tab_drawdown, tab_size = st.tabs([
        "🗓️ Activity Heatmap", "⏱️ P/L Timeline", "📉 Drawdown", "📦 Trade Size Distribution"
    ])

    with tab_heat:
        st.markdown("**Trade frequency heatmap — Day of week vs Hour of day**")
        df2 = df.copy()
        df2["dow"]  = df2["timestamp"].dt.day_name()
        df2["hour"] = df2["timestamp"].dt.hour
        pivot = df2.groupby(["dow", "hour"]).size().reset_index(name="count")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot["dow"] = pd.Categorical(pivot["dow"], categories=days_order, ordered=True)
        heat_matrix = pivot.pivot_table(index="dow", columns="hour", values="count", fill_value=0)
        fig_heat = px.imshow(
            heat_matrix, labels=dict(x="Hour of Day", y="Day of Week", color="Trades"),
            color_continuous_scale="YlOrRd", aspect="auto", title="Trading Activity Heatmap",
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(title="Hour of Day", dtick=1))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("🔴 Red clusters = high-frequency windows — cross-reference with your P/L to see if activity correlates with worse outcomes.")

    with tab_time:
        df_t = df.copy().reset_index(drop=True)
        df_t["cumulative_pl"] = df_t["profit_loss"].cumsum()
        df_t["trade_num"]     = range(1, len(df_t) + 1)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_t["trade_num"], y=df_t["cumulative_pl"],
            mode="lines", name="Cumulative P/L",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
        ))
        if rt.get("flagged_trades"):
            ft = pd.DataFrame(rt["flagged_trades"])
            if "timestamp" in ft.columns:
                ft["timestamp"] = pd.to_datetime(ft["timestamp"])
                merged = df_t.merge(ft[["timestamp"]], on="timestamp", how="inner")
                if not merged.empty:
                    fig_cum.add_trace(go.Scatter(
                        x=merged["trade_num"], y=merged["cumulative_pl"],
                        mode="markers", name="Revenge Trade",
                        marker=dict(color="#ef4444", size=12, symbol="x"),
                    ))
        fig_cum.update_layout(
            title="Cumulative P/L (trade-by-trade)", xaxis_title="Trade Number",
            yaxis_title="Cumulative P/L ($)", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        df_t["hour"] = df_t["timestamp"].dt.hour
        hourly_pl    = df_t.groupby("hour")["profit_loss"].mean().reset_index()
        hourly_pl["color"] = hourly_pl["profit_loss"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
        fig_hr = go.Figure(go.Bar(x=hourly_pl["hour"], y=hourly_pl["profit_loss"],
                                  marker_color=hourly_pl["color"], name="Avg P/L per Hour"))
        fig_hr.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_hr.update_layout(title="Average P/L by Hour of Day", xaxis=dict(title="Hour", dtick=1),
                              yaxis_title="Avg P/L ($)", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hr, use_container_width=True)

    with tab_drawdown:
        df_d = df.copy()
        df_d["peak"]     = df_d["balance"].cummax()
        df_d["drawdown"] = (df_d["balance"] - df_d["peak"]) / df_d["peak"] * 100
        max_dd = df_d["drawdown"].min()

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_d["timestamp"], y=df_d["drawdown"],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.18)",
            line=dict(color="#ef4444"), name="Drawdown %",
        ))
        fig_dd.update_layout(title=f"Account Drawdown (Max: {max_dd:.1f}%)",
                              yaxis_title="Drawdown (%)", xaxis_title="Time",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dd, use_container_width=True)

        if max_dd < -10:
            st.warning(f"⚠️ Your maximum drawdown is **{max_dd:.1f}%**. Drawdowns beyond 10% significantly increase the psychological pressure that drives revenge trading and overtrading.")
        else:
            st.success(f"✅ Max drawdown: **{max_dd:.1f}%** — within manageable range.")

    with tab_size:
        avg_qty = df["quantity"].mean()
        fig_size = px.histogram(df, x="quantity", nbins=40,
                                title="Distribution of Trade Sizes (Quantity)",
                                color_discrete_sequence=["#6366f1"])
        fig_size.add_vline(x=avg_qty, line_dash="dash", line_color="gold",
                           annotation_text=f"Avg: {avg_qty:.2f}", annotation_position="top right")
        fig_size.add_vline(x=avg_qty * 1.5, line_dash="dot", line_color="#ef4444",
                           annotation_text="1.5× (Revenge threshold)", annotation_position="top right")
        fig_size.update_layout(xaxis_title="Quantity", yaxis_title="Count",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_size, use_container_width=True)
        outlier_count = len(df[df["quantity"] > avg_qty * 1.5])
        st.info(f"**{outlier_count}** trades ({outlier_count/len(df)*100:.1f}%) exceeded 1.5× your average size — potential revenge or impulsive trades.")

    # ── SECTION 4: Personalised Suggestions ─────────────────
    st.markdown('<p class="section-head">🎯 Personalised Suggestions</p>', unsafe_allow_html=True)

    with st.expander("📏 Daily Trade Limit", expanded=True):
        suggested_daily = max(int(avg_hourly * 6), 5)
        tighter_daily   = max(int(avg_hourly * 4), 3)
        st.markdown(f"""
        <div class="insight-box">
        Based on your average hourly rate of <strong>{avg_hourly:.1f} trades/hour</strong>, we recommend:
        <ul style="margin-top:0.5rem;margin-bottom:0;">
          <li><strong>Conservative limit:</strong> {tighter_daily} trades/day — forces highest-quality setups only.</li>
          <li><strong>Standard limit:</strong> {suggested_daily} trades/day — allows active trading while discouraging noise trades.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Recommended Daily Limit",  f"{suggested_daily} trades")
        col_b.metric("Conservative Daily Limit", f"{tighter_daily} trades")

    with st.expander("🛑 Stop-Loss Discipline", expanded=la["flagged"]):
        suggested_rr = max(round(loss_ratio * 0.6, 1), 1.5) if avg_win > 0 else 1.5
        st.markdown(f"""
        <div class="insight-box">
        Your current avg loss is <strong>${avg_loss:.2f}</strong> vs avg win of <strong>${avg_win:.2f}</strong>
        (ratio: <strong>{loss_ratio:.2f}×</strong>).
        {'This exceeds the healthy 1.5× threshold — disciplined stop-losses are urgent.' if la["flagged"] else 'Maintain this by pre-setting stops on every trade.'}
        </div>
        """, unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.metric("Target Risk/Reward",       f"1 : {suggested_rr}")
        r2.metric("Suggested Max Loss/Trade", f"${min(avg_loss * 0.6, avg_win) if avg_win > 0 else 0:.2f}")
        r3.metric("Current Loss Ratio",       f"{loss_ratio:.2f}×")
        st.markdown("""
        <div class="rec-card" style="border-color:#c62828;">
            <h4>📌 Stop-Loss Rules to Implement Now</h4>
            <p>
            1. <strong>Set your stop BEFORE you enter</strong> — never after.<br>
            2. <strong>Never move your stop further away</strong> — only closer as the trade moves in your favour.<br>
            3. Use a <strong>hard monetary stop</strong> (e.g. max $X loss per trade) in addition to a technical stop.<br>
            4. If you feel the urge to "give the trade more room," that is a loss aversion signal — respect your original stop.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("❄️ Cooling-Off Periods", expanded=rt["flagged"]):
        st.markdown(f"""
        <div class="insight-box">
        {'<strong>⚠️ Revenge trading detected.</strong> You opened oversized positions after losses ' + str(revenge_count) + ' time(s). A mandatory cooling-off protocol is strongly recommended.' if rt["flagged"] else '✅ No revenge trades detected — maintain this by keeping a cooling-off habit after any loss.'}
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="gauge-card"><div class="gauge-title">After a loss</div><div class="gauge-val" style="color:#f59e0b;">30</div><div class="gauge-badge" style="color:#f59e0b;">min cooldown</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="gauge-card"><div class="gauge-title">After 3 losses in a row</div><div class="gauge-val" style="color:#ef4444;">180</div><div class="gauge-badge" style="color:#ef4444;">min cooldown</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="gauge-card"><div class="gauge-title">After daily limit hit</div><div class="gauge-val" style="color:#8b5cf6;">OFF</div><div class="gauge-badge" style="color:#8b5cf6;">close platform</div></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-card" style="border-color:#0369a1;">
            <h4>❄️ Cooling-Off Protocol — Step by Step</h4>
            <p>
            1. After any loss: <strong>step away from your screen</strong> for at least 30 minutes.<br>
            2. During cooldown: do a physical reset — walk, stretch, breathe. Do <em>not</em> watch charts.<br>
            3. Before re-entering: run through your pre-trade checklist. If you can't tick every box, do not trade.<br>
            4. If you feel "I <em>need</em> to make this back" — that is revenge trading instinct. Add another 30 minutes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("⏱️ Frequency Control", expanded=ot["flagged"]):
        best_hours = []
        if not hourly.empty:
            df_t3 = df.copy()
            df_t3["hour"] = df_t3["timestamp"].dt.hour
            hour_pl = df_t3.groupby("hour")["profit_loss"].mean()
            best_hours = hour_pl.nlargest(3).index.tolist()
        st.markdown(f"""
        <div class="insight-box">
        {'<strong>Your best-performing hours</strong> based on average P/L are: <strong>' + ', '.join([f"{h:02d}:00" for h in best_hours]) + '</strong>. Consider concentrating your trading in these windows.' if best_hours else 'Upload more data to identify your best-performing hours.'}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="rec-card" style="border-color:#d97706;">
            <h4>⏱️ Frequency Control Rules</h4>
            <p>
            1. <strong>Define your trading window</strong> — pick 1–2 sessions per day and stay out otherwise.<br>
            2. Use a <strong>pre-trade checklist</strong>: setup present? Volume confirmed? News risk checked?<br>
            3. <strong>Quality over quantity</strong> — five high-conviction trades beat twenty noise trades every time.<br>
            4. Track your win rate separately for planned vs impulse trades — the data will convince you faster than any rule.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 5: Journaling Prompts ───────────────────────
    st.markdown('<p class="section-head">📓 Trading Psychology Journal Prompts</p>', unsafe_allow_html=True)
    st.markdown("Use these after each session. Write freely — the goal is pattern recognition across weeks.")

    prompt_tabs = st.tabs(["🔄 Overtrading", "😰 Loss Aversion", "😤 Revenge Trading", "🌅 Daily Reflection"])

    with prompt_tabs[0]:
        for label, q in [
            ("Before the session", "What was my plan going in? How many trades was I expecting to place, and why?"),
            ("After the session",  "Did I take any trades where I couldn't immediately articulate a clear reason? What was I feeling when I entered?"),
            ("Pattern recognition","What time of day did most of my trades cluster? Was I bored, anxious, or excited during those periods?"),
            ("FOMO check",         "Which trades did I take because I was afraid of missing a move — not because my setup was present?"),
        ]:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[1]:
        for label, q in [
            ("On losing trades",       "At what point did I first think about closing this trade? What stopped me from acting on that instinct?"),
            ("On winners exited early","I closed this winner early. What would have happened if I had stayed in to my original target?"),
            ("Anchoring check",        "Am I holding because the trade has merit, or because I'm waiting to break even on my entry price?"),
            ("Scenario flip",          "If I had no position right now, would I open this trade at the current price? If not — why am I holding it?"),
        ]:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[2]:
        for label, q in [
            ("After a loss",              "Rate my emotional state after this loss: 1–10. At what number am I safe to trade again?"),
            ("Pre-trade check",           "Am I placing this trade because the setup is valid, or because I lost money earlier and want to recover it?"),
            ("Size check",                "Is this trade larger than my usual size? If yes — is it based on a larger edge, or emotion?"),
            ("Consequence visualisation", "If this trade also loses, what will I feel? Am I comfortable with that outcome?"),
        ]:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[3]:
        for label, q in [
            ("Session summary",   "In one sentence: was today a process-driven day or an outcome-driven day?"),
            ("Best decision",     "What was the best decision I made today — not necessarily the most profitable, but the most disciplined?"),
            ("One thing to change","If I could replay today, what is the single thing I would do differently?"),
            ("Gratitude & growth","What did the market teach me today that I didn't know when I woke up?"),
        ]:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    # ── SECTION 6: Live Journal Entry ───────────────────────
    st.markdown('<p class="section-head">✍️ Log a Journal Entry</p>', unsafe_allow_html=True)
    st.markdown("Record your reflections directly in the app. Entries are stored for the session.")

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    with st.form("journal_form"):
        j_date    = st.date_input("Session date", value=datetime.today())
        j_mood    = st.select_slider("Emotional state before session",
                                     options=["Very Calm", "Calm", "Neutral", "Anxious", "Very Anxious"],
                                     value="Neutral")
        j_plan    = st.text_area("What was your pre-session plan?",
                                 placeholder="E.g. Max 3 trades, only BTC, wait for support level confirmation...")
        j_debrief = st.text_area("Post-session debrief",
                                 placeholder="What happened? Did you stick to the plan?")
        j_bias    = st.multiselect("Which biases showed up today (self-assessed)?",
                                   ["Overtrading", "Loss Aversion", "Revenge Trading", "FOMO", "None"])
        j_lesson  = st.text_area("One lesson from today",
                                 placeholder="e.g. I need to wait 10 min after entry before checking P/L...")
        submit    = st.form_submit_button("💾 Save Entry", type="primary")

    if submit:
        st.session_state.journal_entries.append({
            "date":    str(j_date),
            "mood":    j_mood,
            "plan":    j_plan,
            "debrief": j_debrief,
            "biases":  ", ".join(j_bias) if j_bias else "None",
            "lesson":  j_lesson,
        })
        st.success("✅ Journal entry saved!")

    if st.session_state.journal_entries:
        st.markdown("**Previous Entries This Session:**")
        st.dataframe(pd.DataFrame(st.session_state.journal_entries), use_container_width=True, hide_index=True)

    st.divider()
    st.caption("NBC Bias Detector · Feedback & Recommendations · Educational purposes only — not financial advice.")


# ═════════════════════════════════════════════════════════════
# PAGE: LEARNING CENTRE
# ═════════════════════════════════════════════════════════════
elif page == "📚 Learning Centre":

    st.markdown("""
    <style>
    .section-banner { padding: 1.4rem 2rem; border-radius: 14px 14px 0 0; margin-bottom: 0; }
    .banner-ot { background: linear-gradient(90deg, #f9a825 0%, #ffcc02 100%); }
    .banner-la { background: linear-gradient(90deg, #c62828 0%, #e57373 100%); }
    .banner-rt { background: linear-gradient(90deg, #283593 0%, #5c6bc0 100%); }
    .banner-title    { font-size: 1.6rem; font-weight: 800; color: white; margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,0.2); }
    .banner-subtitle { color: rgba(255,255,255,0.88); font-size: 0.95rem; margin: 0.3rem 0 0 0; }
    .info-quote { background: #f8f9fa; border-left: 5px solid #adb5bd; padding: 1rem 1.4rem;
                  border-radius: 0 8px 8px 0; font-style: italic; font-size: 1rem; color: #495057; margin: 1rem 0; }
    .info-tip  { background: #d4edda; border: 1px solid #28a745; border-radius: 8px;
                 padding: 1rem 1.4rem; color: #155724; font-size: 0.95rem; margin: 1rem 0; }
    .info-warn { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
                 padding: 1rem 1.4rem; color: #856404; font-size: 0.95rem; margin: 1rem 0; }
    .quiz-banner { background: linear-gradient(90deg, #1a237e, #3949ab); border-radius: 14px;
                   padding: 2rem 2.5rem; margin-bottom: 1.5rem; text-align: center; }
    .quiz-banner h2 { color: white; font-size: 2rem; margin: 0 0 0.4rem; }
    .quiz-banner p  { color: rgba(255,255,255,0.85); margin: 0; font-size: 1rem; }
    .score-box  { text-align: center; padding: 2rem; border-radius: 14px; margin: 1rem 0; }
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

    section = st.radio(
        "Jump to section:",
        ["🔄 Overtrading", "😰 Loss Aversion", "😤 Revenge Trading", "📋 Quick Reference", "🧠 Quiz"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    # ── OVERTRADING ──────────────────────────────────────────
    if section == "🔄 Overtrading":
        st.markdown('<div class="section-banner banner-ot"><p class="banner-title">🔄 Overtrading</p><p class="banner-subtitle">Bias 01 — Trading too frequently or with too much size relative to your account</p></div>', unsafe_allow_html=True)
        t1, t2, t3, t4, t5 = st.tabs(["📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"])
        with t1:
            st.markdown("### What is Overtrading?")
            st.markdown("Overtrading means executing **far more trades than a sound strategy justifies**. It comes in three flavours that often overlap:")
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
            st.markdown('<div class="info-quote">"A disciplined trader fires only when their edge is clearly present. Overtraders fire any time they <em>feel</em> like something might happen."</div>', unsafe_allow_html=True)
        with t2:
            st.markdown("### Why Does Overtrading Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**⚡ Action Bias**"); st.markdown("Humans are wired to *do something* when anxious. Sitting in cash feels like losing even when it's the correct position.")
                with st.container(border=True):
                    st.markdown("**😰 FOMO**"); st.markdown("Every price wiggle looks like a missed opportunity. The brain exaggerates the cost of inaction.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🎰 Dopamine Loops**"); st.markdown("Placing orders triggers the same reward circuits as gambling. The act of trading feels good independently of the outcome.")
                with st.container(border=True):
                    st.markdown("**😴 Boredom**"); st.markdown("Slow markets cause traders to manufacture setups that simply aren't there.")
            st.markdown('<div class="info-quote">"The market is a device for transferring money from the impatient to the patient." — Warren Buffett</div>', unsafe_allow_html=True)
        with t3:
            st.markdown("### What Does Overtrading Cost?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**Direct Costs**"); st.markdown("- Spreads and commissions compound with every unnecessary trade\n- Slippage increases when entries are impulsive (bad fills)\n- Costs accumulate whether the trade wins or loses")
            with col2:
                with st.container(border=True):
                    st.markdown("**Indirect Costs**"); st.markdown("- Cognitive fatigue degrades decision quality mid-session\n- Overexposure — many open positions means one market move wipes several at once\n- Drift from strategy — impulsive trades break your tested edge")
            st.markdown('<div class="info-warn">⚠️ <b>Real cost example:</b> A trader making 20 trades/day at $5 commission = $100/day in fees alone — that\'s <b>$26,000/year</b> before a single market move is even considered.</div>', unsafe_allow_html=True)
        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**Maya's morning session** — Account: $10,000")
                for time_str, action, tag in [
                    ("9:30 AM", "Buys 0.5 BTC", "Entry"),
                    ("9:38 AM", "Price dips — panic sells, immediately re-buys", "Switch"),
                    ("9:45 AM", "Buys ETH — price flat — sells 7 min later", "Churn"),
                    ("9:52 AM", "Re-enters BTC — exits 6 min later", "Churn"),
                    ("10:00 AM", "8 trades completed in 30 minutes", "Result"),
                ]:
                    c1, c2, c3 = st.columns([1, 4, 1])
                    c1.markdown(f"`{time_str}`"); c2.markdown(action)
                    c3.markdown("🔴" if tag in ("Switch", "Churn") else ("🏁" if tag == "Result" else "⚪"))
            st.markdown('<div class="info-warn">Maya\'s account is down <b>1.2%</b> — not because the market moved against her, but purely from transaction costs and bad fills on impulsive entries.</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-tip">✅ <b>The Fix:</b> Set a hard daily trade limit (e.g. max 5 trades/day). Only enter when every item on your pre-trade checklist is met — not just because it "feels right."</div>', unsafe_allow_html=True)
        with t5:
            st.markdown("### How `bias_engine.py` Detects Overtrading")
            st.info("Three independent sub-checks run in parallel — **any one** can flag the bias.", icon="ℹ️")
            with st.expander("📊 Sub-check A — Trades per hour", expanded=True):
                st.code("""
hourly   = df.set_index("timestamp").resample("1h").size()
peak_val = int(hourly.max())
total_hours  = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600
avg_hourly   = len(df) / total_hours
max_per_hour = max(int(avg_hourly * 2), 15)
if peak_val > max_per_hour:
    result["flagged"] = True
    result["reasons"].append(f"Executed {peak_val} trades in one hour (threshold: {max_per_hour})")
                """, language="python")
            with st.expander("📊 Sub-check B — Volume / balance ratio"):
                st.code("""
df["trade_value"] = df["quantity"] * df["entry_price"]
ratio = df["trade_value"].sum() / df["balance"].mean()
if ratio > max_vol_ratio:
    result["flagged"] = True
    result["reasons"].append(f"Total volume is {ratio:.1f}× your average balance")
                """, language="python")
            with st.expander("📊 Sub-check C — Rapid position switching"):
                st.code("""
switches = 0
for asset in df["asset"].unique():
    sub = df[df["asset"] == asset].reset_index(drop=True)
    for i in range(1, len(sub)):
        diff_min = (sub.loc[i,"timestamp"] - sub.loc[i-1,"timestamp"]).total_seconds()/60
        if sub.loc[i,"buy_sell"] != sub.loc[i-1,"buy_sell"] and diff_min < 30:
            switches += 1
if switches >= 3:
    result["flagged"] = True
    result["reasons"].append(f"Detected {switches} rapid position switches")
                """, language="python")

    # ── LOSS AVERSION ────────────────────────────────────────
    elif section == "😰 Loss Aversion":
        st.markdown('<div class="section-banner banner-la"><p class="banner-title">😰 Loss Aversion</p><p class="banner-subtitle">Bias 02 — Holding losing trades too long while cutting winners short</p></div>', unsafe_allow_html=True)
        t1, t2, t3, t4, t5 = st.tabs(["📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"])
        with t1:
            st.markdown("### What is Loss Aversion?")
            st.markdown("Loss aversion is the cognitive bias where the **pain of losing feels roughly twice as powerful** as the pleasure of an equivalent gain *(Kahneman & Tversky, 1979)*.")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("#### ✂️ Cutting Winners Short"); st.markdown("Closing profitable trades too early out of fear. Result: **small average wins**.")
            with col2:
                with st.container(border=True):
                    st.markdown("#### 🪝 Riding Losers Long"); st.markdown("Refusing to close a losing trade while hoping it will recover. Result: **large average losses**.")
            st.markdown('<div class="info-quote">"An open losing trade isn\'t officially a loss yet. Closing it makes it real — and loss aversion makes that reality feel unbearable."</div>', unsafe_allow_html=True)
        with t2:
            st.markdown("### Why Does Loss Aversion Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**🛡️ Ego Protection**"); st.markdown("An open losing trade isn't officially a loss yet. Closing it makes it real and forces you to confront the mistake.")
                with st.container(border=True):
                    st.markdown("**⚓ Anchoring**"); st.markdown("Traders anchor to their entry price and wait for 'break even' — even as the loss compounds further below.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🕳️ Sunk-Cost Fallacy**"); st.markdown('"I\'ve already lost $500 — I can\'t close now." Past losses irrationally influence future decisions.')
                with st.container(border=True):
                    st.markdown("**📉 Prospect Theory**"); st.markdown("Our brains weigh losses 2× more than gains — causing asymmetric behaviour.")
            st.markdown('<div class="info-quote">"The first loss is the best loss." — Trading floor adage</div>', unsafe_allow_html=True)
        with t3:
            st.markdown("### The Mathematical Damage")
            st.dataframe({
                "Win Rate": ["50%", "60%", "70%", "80%"],
                "Expected Value Per Trade": ["−$60.00", "−$32.00", "−$4.00", "+$24.00"],
                "Result After 100 Trades": ["−$6,000", "−$3,200", "−$400", "+$2,400"],
                "Verdict": ["🔴 Losing", "🔴 Losing", "🔴 Losing", "🟢 Profitable"],
            }, use_container_width=True, hide_index=True)
            st.markdown('<div class="info-warn">⚠️ Most retail traders have a win rate below 60%. With a 2.5× loss ratio they are <b>mathematically guaranteed to lose money in the long run</b>.</div>', unsafe_allow_html=True)
        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**James's trading day** — Asset: ETH")
                for time_str, action, tag in [
                    ("10:00", "Buys ETH at $3,000", "neutral"),
                    ("10:45", "ETH rises to $3,120 (+4%). James fears a reversal — exits. Profit: +$120", "win"),
                    ("11:00", "ETH continues to $3,300. James re-enters at $3,280", "neutral"),
                    ("13:30", "ETH drops to $2,950. James thinks 'it'll bounce' — holds", "loss"),
                    ("16:00", "ETH drops to $2,700. James finally closes. Loss: −$580", "loss"),
                ]:
                    c1, c2 = st.columns([1, 5])
                    c1.markdown(f"`{time_str}`")
                    c2.markdown(f"{'🟢' if tag=='win' else ('🔴' if tag=='loss' else '⚪')} {action}")
            st.markdown('<div class="info-tip">✅ <b>The Fix:</b> Set a hard stop-loss AND a profit target before every trade. Use a minimum 1:1.5 risk/reward ratio. <b>Never move your stop-loss further away after entry.</b></div>', unsafe_allow_html=True)
        with t5:
            st.markdown("### How `bias_engine.py` Detects Loss Aversion")
            with st.expander("📊 Sub-check A — Win/Loss P/L size ratio", expanded=True):
                st.code("""
avg_win  = df[df["profit_loss"] > 0]["profit_loss"].mean()
avg_loss = df[df["profit_loss"] < 0]["profit_loss"].abs().mean()
ratio    = avg_loss / avg_win
if ratio > ratio_threshold:   # default: 1.5×
    result["flagged"] = True
    result["reasons"].append(f"Average loss (${avg_loss:.2f}) is {ratio:.1f}× your average win")
                """, language="python")
            with st.expander("📊 Sub-check B — Price range asymmetry"):
                st.code("""
df["price_range"]  = (df["exit_price"] - df["entry_price"]).abs()
avg_range_win  = df[df["profit_loss"] > 0]["price_range"].mean()
avg_range_loss = df[df["profit_loss"] < 0]["price_range"].mean()
range_ratio    = avg_range_loss / avg_range_win
if range_ratio > price_range_ratio:   # default: 1.5×
    result["flagged"] = True
    result["reasons"].append(f"Losing trades travel {range_ratio:.1f}× more price distance before exit")
                """, language="python")

    # ── REVENGE TRADING ──────────────────────────────────────
    elif section == "😤 Revenge Trading":
        st.markdown('<div class="section-banner banner-rt"><p class="banner-title">😤 Revenge Trading</p><p class="banner-subtitle">Bias 03 — Opening oversized positions right after a loss to "win back" money</p></div>', unsafe_allow_html=True)
        t1, t2, t3, t4, t5 = st.tabs(["📖 What is it?", "🧠 Why it happens", "💸 What it costs", "📘 Example", "🔬 How we detect it"])
        with t1:
            st.markdown("### What is Revenge Trading?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("#### ⚡ Timing"); st.markdown("The new trade opens **very shortly** after the losing trade closes — before any rational analysis can take place.")
            with col2:
                with st.container(border=True):
                    st.markdown("#### 📏 Size"); st.markdown("The new position is **significantly larger** than the trader's historical average — magnifying risk at the worst emotional moment.")
            st.markdown('<div class="info-quote">"The market does not know or care that you just lost money — and it will not give it back on demand."</div>', unsafe_allow_html=True)
        with t2:
            st.markdown("### Why Does Revenge Trading Happen?")
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown("**🎲 Gambler's Fallacy**"); st.markdown('"I just lost, so I\'m *due* a win." The market has no memory.')
                with st.container(border=True):
                    st.markdown("**🪞 Ego Threat**"); st.markdown("A loss feels like personal failure. A quick recovery trade would restore self-image.")
            with col2:
                with st.container(border=True):
                    st.markdown("**🧠 Emotional Hijacking**"); st.markdown("The amygdala fires after a loss and overrides the prefrontal cortex — decisions become reactive, not analytical.")
                with st.container(border=True):
                    st.markdown("**📌 Recency Bias**"); st.markdown("The last trade feels more significant than the statistical average, distorting risk assessment.")
            st.markdown('<div class="info-quote">"After a loss, the worst thing you can do is try to make it back immediately." — Mark Douglas, <em>Trading in the Zone</em></div>', unsafe_allow_html=True)
        with t3:
            st.markdown("### The Compounding Damage")
            st.dataframe({
                "Trade": ["Normal loss", "Revenge trade (3× size)", "Total damage"],
                "Position Size": ["1% of account", "3% of account", "—"],
                "P/L": ["−$100", "−$300", "−$400"],
                "vs. Stopping After Loss": ["−$100", "—", "4× worse"],
            }, use_container_width=True, hide_index=True)
        with t4:
            st.markdown("### Real-World Example")
            with st.container(border=True):
                st.markdown("**Sofia's afternoon** — Normal position size: 0.3 BTC")
                for time_str, action, tag in [
                    ("2:10 PM", "Buys 0.3 BTC at $42,000. Price falls.", "neutral"),
                    ("2:12 PM", "Exits at $41,500. Loss: −$150", "loss"),
                    ("2:14 PM", "Angry — buys 0.9 BTC (3× normal size) at $41,500 to 'get it back'", "revenge"),
                    ("2:28 PM", "Price falls to $40,800. Exits. Loss: −$630", "loss"),
                    ("2:28 PM", "Total damage in 18 minutes: −$780 instead of −$150", "result"),
                ]:
                    c1, c2 = st.columns([1, 5])
                    c1.markdown(f"`{time_str}`")
                    c2.markdown(f"{'🔴' if tag=='loss' else ('🚨' if tag=='revenge' else ('💥' if tag=='result' else '⚪'))} {action}")
            st.markdown('<div class="info-tip">✅ <b>The Fix:</b> Mandatory 30-minute cooling-off rule after any loss. Step away from the screen. Only re-enter if a pre-defined setup is present.</div>', unsafe_allow_html=True)
        with t5:
            st.markdown("### How `bias_engine.py` Detects Revenge Trading")
            st.info("Uses Pandas `.shift()` to compare each trade against the one immediately before it — no slow Python loops.", icon="ℹ️")
            with st.expander("📊 Full detection logic", expanded=True):
                st.code("""
df      = df.sort_values("timestamp").reset_index(drop=True)
avg_qty = df["quantity"].mean()

df["prev_pl"]         = df["profit_loss"].shift(1)
df["prev_ts"]         = df["timestamp"].shift(1)
df["time_since_prev"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60

mask = (
    (df["prev_pl"] < 0)
    & (df["quantity"] > avg_qty * qty_multiplier)   # default: 1.5×
    & (df["time_since_prev"] <= time_window_min)     # default: 15 min
)
flagged_df = df[mask]
                """, language="python")

    # ── QUICK REFERENCE ──────────────────────────────────────
    elif section == "📋 Quick Reference":
        st.markdown("## 📋 Quick Reference — All Three Biases")
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
                    for label, val in [("Core behaviour", behaviour), ("Emotional driver", driver),
                                       ("CSV fields used", fields), ("Pandas technique", technique),
                                       ("Detection threshold", threshold)]:
                        st.markdown(f"**{label}**"); st.markdown(val)
                    st.markdown(f'<div class="info-tip">✅ <b>Quick fix:</b> {fix}</div>', unsafe_allow_html=True)
        st.divider()
        st.caption("NBC Bias Detector · Learning Centre · Educational only, not financial advice.")

    # ── QUIZ ─────────────────────────────────────────────────
    elif section == "🧠 Quiz":
        st.markdown("""
        <div class="quiz-banner">
            <h2>🧠 Knowledge Quiz</h2>
            <p>Test your understanding of the three trading biases · 10 questions · Instant feedback</p>
        </div>
        """, unsafe_allow_html=True)

        questions = [
            {"q": "A trader executes 18 trades in a single hour. Which bias does this most directly indicate?",
             "options": ["Loss Aversion", "Revenge Trading", "Overtrading", "Anchoring Bias"],
             "answer": "Overtrading",
             "explanation": "Time-based clustering — too many trades in a single hour — is the primary signal of **Overtrading**. Our engine flags this using `resample('1h')`."},
            {"q": "Sarah's average winning trade returns $90, but her average losing trade costs her $220. What is her loss/win ratio and what does it indicate?",
             "options": ["0.41× — she is managing risk well", "2.44× — she likely has Loss Aversion", "1.0× — perfectly balanced", "2.44× — she likely has Revenge Trading tendencies"],
             "answer": "2.44× — she likely has Loss Aversion",
             "explanation": "Loss/win ratio = 220 ÷ 90 = **2.44×**. Our threshold is 1.5×. A ratio above 1.5× indicates **Loss Aversion**."},
            {"q": "In `bias_engine.py`, which Pandas method is used to compare each trade against the immediately preceding one without using a for-loop?",
             "options": ["`df.merge()`", "`df.shift(1)`", "`df.rolling(1)`", "`df.diff(1)`"],
             "answer": "`df.shift(1)`",
             "explanation": "`.shift(1)` moves the entire column down by one row in a single vectorised operation."},
            {"q": "Tom loses $200 on a trade at 2:05 PM. At 2:09 PM he opens a new position at 3× his normal size. What bias is this?",
             "options": ["Overtrading", "Loss Aversion", "Confirmation Bias", "Revenge Trading"],
             "answer": "Revenge Trading",
             "explanation": "A significantly oversized trade opened within minutes of a loss is the textbook definition of **Revenge Trading**."},
            {"q": "Which psychological theory explains why a $100 loss feels roughly as painful as a $200 gain feels good?",
             "options": ["Efficient Market Hypothesis", "Prospect Theory (Kahneman & Tversky)", "Random Walk Theory", "Modern Portfolio Theory"],
             "answer": "Prospect Theory (Kahneman & Tversky)",
             "explanation": "**Prospect Theory** (1979) established that losses loom approximately 2× larger than equivalent gains in human subjective experience."},
            {"q": "The NBC Bias Detector auto-calculates the overtrading threshold as 2× your average hourly trade rate. If you average 4 trades/hour, what threshold is set?",
             "options": ["4", "6", "8", "15"],
             "answer": "8",
             "explanation": "2 × 4 = **8** trades/hour. There is a minimum of 15 — but 8 is below 15, so the threshold would actually be capped at 15."},
            {"q": "A trader rapidly buys and sells the same asset three times in 20 minutes, alternating direction each time. Which overtrading sub-check catches this?",
             "options": ["Sub-check A — Trades per hour", "Sub-check B — Volume / balance ratio", "Sub-check C — Rapid position switching", "None — this is not overtrading"],
             "answer": "Sub-check C — Rapid position switching",
             "explanation": "**Sub-check C** specifically looks for Buy→Sell→Buy flips on the same asset within a 30-minute window."},
            {"q": "Emma has a 65% win rate but is still losing money overall. What is the most likely explanation?",
             "options": ["Her win rate calculation is wrong", "She is experiencing Overtrading — too many commission costs", "Her average loss is significantly larger than her average win (Loss Aversion)", "She is trading the wrong assets"],
             "answer": "Her average loss is significantly larger than her average win (Loss Aversion)",
             "explanation": "A 65% win rate with avg win $80 and avg loss $200 = `0.65×80 − 0.35×200 = −$18/trade`. **Loss Aversion** creates a negative expected value even at above-average win rates."},
            {"q": "Which column in the CSV does the revenge trading detector use to measure the gap between a loss and the next trade?",
             "options": ["`profit_loss`", "`entry_price`", "`timestamp`", "`balance`"],
             "answer": "`timestamp`",
             "explanation": "After `.shift(1)`, the engine calculates `(current_timestamp - prev_timestamp).dt.total_seconds() / 60` to get the gap in minutes."},
            {"q": "Which of these is NOT one of the three biases detected by the NBC Bias Detector?",
             "options": ["Overtrading", "Confirmation Bias", "Loss Aversion", "Revenge Trading"],
             "answer": "Confirmation Bias",
             "explanation": "The NBC Bias Detector monitors **Overtrading**, **Loss Aversion**, and **Revenge Trading**. Confirmation Bias is real but not detected by this tool."},
        ]

        if "quiz_answers" not in st.session_state:
            st.session_state.quiz_answers = {}
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False

        if not st.session_state.quiz_submitted:
            for i, q in enumerate(questions):
                with st.container(border=True):
                    st.markdown(f"**Question {i+1} of {len(questions)}**")
                    st.markdown(f"#### {q['q']}")
                    answer = st.radio(f"q{i}", q["options"], index=None, label_visibility="collapsed", key=f"quiz_q{i}")
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
            score = sum(1 for i, q in enumerate(questions) if st.session_state.quiz_answers.get(i) == q["answer"])
            pct   = score / len(questions) * 100
            if pct >= 80:   box_class, emoji, verdict = "score-great", "🏆", "Excellent! You have a strong grasp of trading psychology."
            elif pct >= 60: box_class, emoji, verdict = "score-ok",    "📈", "Good effort! Review the sections where you made mistakes."
            else:           box_class, emoji, verdict = "score-low",   "📚", "Keep studying! Head back through the Learning Centre sections."

            st.markdown(f'<div class="score-box {box_class}"><h1>{emoji}</h1><h2>{score} / {len(questions)} correct ({pct:.0f}%)</h2><p>{verdict}</p></div>', unsafe_allow_html=True)
            st.markdown("### Detailed Feedback")
            for i, q in enumerate(questions):
                user_ans = st.session_state.quiz_answers.get(i, "Not answered")
                correct  = user_ans == q["answer"]
                with st.container(border=True):
                    st.markdown(f"{'✅' if correct else '❌'} **Q{i+1}: {q['q']}**")
                    if not correct:
                        st.markdown(f"Your answer: ~~{user_ans}~~")
                    st.markdown(f"**Correct answer: {q['answer']}**")
                    st.info(q["explanation"], icon="💡")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Retake Quiz", use_container_width=True):
                    st.session_state.quiz_answers  = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()