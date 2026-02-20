"""
feedback_recommendations.py
────────────────────────────────────────────────────────────────────────────
NBC Bias Detector — Feedback & Recommendations Page
Drop this file into the same folder as app.py, then add the page to the
sidebar navigation and call render_feedback_page(df, biases) from app.py.
────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — severity badge
# ─────────────────────────────────────────────────────────────────────────────
def _severity(score: float) -> tuple[str, str]:
    """Return (label, hex_colour) for a 0-100 severity score."""
    if score >= 75:
        return "🔴 Critical", "#c62828"
    if score >= 50:
        return "🟠 Moderate", "#e65100"
    if score >= 25:
        return "🟡 Mild", "#f9a825"
    return "🟢 Clear", "#2e7d32"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION  — call this from app.py
# ─────────────────────────────────────────────────────────────────────────────
def render_feedback_page(df: pd.DataFrame, biases: dict):
    """
    Parameters
    ----------
    df     : the full cleaned trading DataFrame (columns as per CSV spec)
    biases : dict returned by bias_engine.run_all()
    """
    ot = biases["overtrading"]
    la = biases["loss_aversion"]
    rt = biases["revenge_trading"]

    # ── Pre-compute stats used across sections ────────────────────────────────
    wins   = df[df["profit_loss"] > 0]["profit_loss"]
    losses = df[df["profit_loss"] < 0]["profit_loss"].abs()

    avg_win    = wins.mean()   if not wins.empty   else 0
    avg_loss   = losses.mean() if not losses.empty else 0
    loss_ratio = avg_loss / avg_win if avg_win > 0 else 0

    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0

    total_hours = max(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600, 1
    )
    avg_hourly = len(df) / total_hours

    revenge_count = rt["details"].get("revenge_trade_count", 0)

    # Compute per-hour trade counts
    hourly = df.set_index("timestamp").resample("1h").size()
    peak_hour_val = int(hourly.max()) if not hourly.empty else 0

    # ── Severity scores (0-100) ───────────────────────────────────────────────
    ot_score = min(100, (peak_hour_val / max(avg_hourly * 2, 1)) * 50) if ot["flagged"] else 10
    la_score = min(100, (loss_ratio / 1.5) * 50)                        if la["flagged"] else 10
    rt_score = min(100, revenge_count * 20)                              if rt["flagged"] else 5

    overall_score = (ot_score + la_score + rt_score) / 3

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE HEADER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    .fb-hero {
        background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 60%, #0d2137 100%);
        border-radius: 20px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255,255,255,0.07);
        position: relative;
        overflow: hidden;
    }
    .fb-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .fb-hero h1 {
        font-family: 'DM Serif Display', serif;
        color: #f0f4f8;
        font-size: 2rem;
        margin: 0 0 0.3rem;
    }
    .fb-hero p { color: rgba(200,220,240,0.7); margin: 0; font-size: 0.95rem; }

    .gauge-card {
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        background: #111d28;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }
    .gauge-title { font-size: 0.8rem; color: #90a4b7; text-transform: uppercase; letter-spacing: 0.08em; }
    .gauge-val   { font-family: 'DM Serif Display', serif; font-size: 2.6rem; margin: 0.2rem 0; }
    .gauge-badge { font-size: 0.85rem; font-weight: 600; }

    .rec-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border-left: 5px solid;
        background: #f8fafc;
    }
    .rec-card h4 { margin: 0 0 0.5rem; font-size: 1rem; }
    .rec-card p  { margin: 0; color: #444; font-size: 0.93rem; line-height: 1.55; }

    .insight-box {
        background: linear-gradient(135deg, #e8f4fd, #dbeafe);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        font-size: 0.93rem;
        color: #1e3a5f;
    }
    .insight-box strong { color: #1d4ed8; }

    .journal-card {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        font-size: 0.93rem;
        color: #78350f;
        line-height: 1.6;
    }
    .heatmap-section { margin: 1.5rem 0; }
    .section-head {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #1a2a3a;
        margin: 2rem 0 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="fb-hero">
        <h1>📋 Feedback &amp; Recommendations</h1>
        <p>Personalised analysis of your trading psychology — actionable suggestions backed by your own data.</p>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — BIAS HEALTH DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
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
    ot_label,      ot_color      = _severity(ot_score)
    la_label,      la_color      = _severity(la_score)
    rt_label,      rt_color      = _severity(rt_score)

    gauge_card(col1, "Overall Risk Score",  overall_score, overall_label, overall_color)
    gauge_card(col2, "🔄 Overtrading",      ot_score,      ot_label,      ot_color)
    gauge_card(col3, "😰 Loss Aversion",    la_score,      la_label,      la_color)
    gauge_card(col4, "😤 Revenge Trading",  rt_score,      rt_label,      rt_color)

    # Spider / Radar chart
    fig_radar = go.Figure(go.Scatterpolar(
        r=[ot_score, la_score, rt_score, ot_score],
        theta=["Overtrading", "Loss Aversion", "Revenge Trading", "Overtrading"],
        fill="toself",
        fillcolor="rgba(220,38,38,0.15)",
        line=dict(color="#ef4444", width=2),
        name="Your Bias Score",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[30, 30, 30, 30],
        theta=["Overtrading", "Loss Aversion", "Revenge Trading", "Overtrading"],
        fill="toself",
        fillcolor="rgba(34,197,94,0.08)",
        line=dict(color="#22c55e", width=1.5, dash="dot"),
        name="Healthy Benchmark",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0f1923",
            radialaxis=dict(range=[0, 100], showticklabels=True, tickfont=dict(color="#94a3b8"), gridcolor="#1e2d3d"),
            angularaxis=dict(tickfont=dict(color="#e2e8f0", size=13), gridcolor="#1e2d3d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#94a3b8")),
        margin=dict(l=60, r=60, t=30, b=30),
        height=340,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — BIAS SUMMARIES (narrative)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">🗣️ Your Bias Summaries</p>', unsafe_allow_html=True)

    # Overtrading summary
    with st.container(border=True):
        st.markdown("#### 🔄 Overtrading")
        if ot["flagged"]:
            st.error(
                f"**You are overtrading.** Your peak hour hit **{peak_hour_val} trades** — "
                f"well above your personalised threshold. "
                f"You average **{avg_hourly:.1f} trades/hour**, suggesting you're frequently entering "
                f"without a high-conviction setup. This pattern is most common during high-volatility "
                f"windows when price action creates the illusion of opportunity on every candle.",
                icon="🚨",
            )
            # Find peak-trading hour label
            if not hourly.empty:
                peak_dt = hourly.idxmax()
                st.info(
                    f"📍 Your highest-frequency hour was **{peak_dt.strftime('%A %d %b, %H:%M')}** "
                    f"with {peak_hour_val} trades. Consider reviewing what market event or emotion "
                    f"drove that session.",
                    icon="🔎",
                )
        else:
            st.success(
                f"**Overtrading: Clear.** Your average of {avg_hourly:.1f} trades/hour is within healthy bounds. "
                f"Keep maintaining discipline on trade frequency.",
                icon="✅",
            )

    # Loss aversion summary
    with st.container(border=True):
        st.markdown("#### 😰 Loss Aversion")
        if la["flagged"]:
            ev_per_trade = (win_rate / 100) * avg_win - (1 - win_rate / 100) * avg_loss
            st.error(
                f"**Loss aversion detected.** Your average loss (${avg_loss:.2f}) is "
                f"**{loss_ratio:.1f}× your average win** (${avg_win:.2f}). "
                f"With a {win_rate:.0f}% win rate, your expected value per trade is "
                f"**${ev_per_trade:+.2f}**. "
                f"{'This is negative — you are losing money in expectation despite winning more than half your trades.' if ev_per_trade < 0 else 'Although positive, the gap between wins and losses creates fragility.'}",
                icon="🚨",
            )
        else:
            st.success(
                f"**Loss Aversion: Clear.** Your win/loss sizing is healthy "
                f"(ratio: {loss_ratio:.2f}×). You're cutting losers and letting winners run appropriately.",
                icon="✅",
            )

    # Revenge trading summary
    with st.container(border=True):
        st.markdown("#### 😤 Revenge Trading")
        if rt["flagged"]:
            st.error(
                f"**Revenge trading detected — {revenge_count} instance(s).** "
                f"After a loss, you opened an oversized position within the alert window. "
                f"These trades are placed at your most emotionally compromised moment, "
                f"with no evidence of edge — amplifying losses exactly when recovery feels most urgent.",
                icon="🚨",
            )
            if rt.get("flagged_trades"):
                flagged_df = pd.DataFrame(rt["flagged_trades"])
                st.dataframe(flagged_df, use_container_width=True, hide_index=True)
        else:
            st.success(
                "**Revenge Trading: Clear.** No oversized positions detected immediately after losses.",
                icon="✅",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — GRAPHICAL INSIGHTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">📊 Graphical Insights</p>', unsafe_allow_html=True)

    tab_heat, tab_time, tab_drawdown, tab_size = st.tabs([
        "🗓️ Activity Heatmap", "⏱️ P/L Timeline", "📉 Drawdown", "📦 Trade Size Distribution"
    ])

    # ── Heatmap: day-of-week × hour-of-day ───────────────────────────────────
    with tab_heat:
        st.markdown("**Trade frequency heatmap — Day of week vs Hour of day**")
        df2 = df.copy()
        df2["dow"]  = df2["timestamp"].dt.day_name()
        df2["hour"] = df2["timestamp"].dt.hour
        pivot = df2.groupby(["dow", "hour"]).size().reset_index(name="count")
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot["dow"] = pd.Categorical(pivot["dow"], categories=days_order, ordered=True)
        pivot = pivot.sort_values(["dow", "hour"])
        heat_matrix = pivot.pivot_table(index="dow", columns="hour", values="count", fill_value=0)
        fig_heat = px.imshow(
            heat_matrix,
            labels=dict(x="Hour of Day", y="Day of Week", color="Trades"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Trading Activity Heatmap",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Hour of Day", dtick=1),
            coloraxis_colorbar=dict(title="# Trades"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("🔴 Red clusters = high-frequency windows — cross-reference with your P/L to see if activity correlates with worse outcomes.")

    # ── P/L colour timeline ───────────────────────────────────────────────────
    with tab_time:
        st.markdown("**Cumulative P/L vs trade-by-trade bar**")
        df_t = df.copy().reset_index(drop=True)
        df_t["cumulative_pl"] = df_t["profit_loss"].cumsum()
        df_t["trade_num"]     = range(1, len(df_t) + 1)

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_t["trade_num"], y=df_t["cumulative_pl"],
            mode="lines", name="Cumulative P/L",
            line=dict(color="#3b82f6", width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        ))
        # Mark revenge trades if any
        if rt.get("flagged_trades"):
            ft = pd.DataFrame(rt["flagged_trades"])
            if "timestamp" in ft.columns:
                ft["timestamp"] = pd.to_datetime(ft["timestamp"])
                merged = df_t.merge(ft[["timestamp"]], on="timestamp", how="inner")
                fig_cum.add_trace(go.Scatter(
                    x=merged["trade_num"], y=merged["cumulative_pl"],
                    mode="markers", name="Revenge Trade",
                    marker=dict(color="#ef4444", size=12, symbol="x"),
                ))
        fig_cum.update_layout(
            title="Cumulative P/L (trade-by-trade)",
            xaxis_title="Trade Number",
            yaxis_title="Cumulative P/L ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # Win/Loss by hour bar
        df_t["hour"] = df_t["timestamp"].dt.hour
        hourly_pl = df_t.groupby("hour")["profit_loss"].mean().reset_index()
        hourly_pl["color"] = hourly_pl["profit_loss"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")
        fig_hr = go.Figure(go.Bar(
            x=hourly_pl["hour"], y=hourly_pl["profit_loss"],
            marker_color=hourly_pl["color"],
            name="Avg P/L per Hour",
        ))
        fig_hr.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_hr.update_layout(
            title="Average P/L by Hour of Day",
            xaxis=dict(title="Hour", dtick=1),
            yaxis_title="Avg P/L ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_hr, use_container_width=True)

    # ── Drawdown chart ────────────────────────────────────────────────────────
    with tab_drawdown:
        st.markdown("**Drawdown from peak balance**")
        df_d = df.copy()
        df_d["peak"]     = df_d["balance"].cummax()
        df_d["drawdown"] = (df_d["balance"] - df_d["peak"]) / df_d["peak"] * 100

        max_dd = df_d["drawdown"].min()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=df_d["timestamp"], y=df_d["drawdown"],
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.18)",
            line=dict(color="#ef4444"),
            name="Drawdown %",
        ))
        fig_dd.update_layout(
            title=f"Account Drawdown (Max: {max_dd:.1f}%)",
            yaxis_title="Drawdown (%)",
            xaxis_title="Time",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        if max_dd < -10:
            st.warning(f"⚠️ Your maximum drawdown is **{max_dd:.1f}%**. Drawdowns beyond 10% significantly increase the psychological pressure that drives revenge trading and overtrading.")
        else:
            st.success(f"✅ Max drawdown: **{max_dd:.1f}%** — within manageable range.")

    # ── Trade size distribution ───────────────────────────────────────────────
    with tab_size:
        st.markdown("**Trade size distribution — spot oversized outliers**")
        avg_qty = df["quantity"].mean()
        fig_size = px.histogram(
            df, x="quantity", nbins=40,
            title="Distribution of Trade Sizes (Quantity)",
            color_discrete_sequence=["#6366f1"],
        )
        fig_size.add_vline(x=avg_qty, line_dash="dash", line_color="gold",
                           annotation_text=f"Avg: {avg_qty:.2f}", annotation_position="top right")
        fig_size.add_vline(x=avg_qty * 1.5, line_dash="dot", line_color="#ef4444",
                           annotation_text="1.5× (Revenge threshold)", annotation_position="top right")
        fig_size.update_layout(
            xaxis_title="Quantity", yaxis_title="Count",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_size, use_container_width=True)

        outlier_count = len(df[df["quantity"] > avg_qty * 1.5])
        st.info(f"**{outlier_count}** trades ({outlier_count/len(df)*100:.1f}%) exceeded 1.5× your average size — these are potential revenge or impulsive trades.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — PERSONALISED SUGGESTIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">🎯 Personalised Suggestions</p>', unsafe_allow_html=True)

    # Daily trade limit recommendation
    with st.expander("📏 Daily Trade Limit", expanded=True):
        suggested_daily = max(int(avg_hourly * 6), 5)   # ~6 active hours, floor of 5
        tighter_daily   = max(int(avg_hourly * 4), 3)

        st.markdown(f"""
        <div class="insight-box">
        Based on your average hourly rate of <strong>{avg_hourly:.1f} trades/hour</strong>, we recommend:
        <ul style="margin-top:0.5rem;margin-bottom:0;">
          <li><strong>Conservative limit:</strong> {tighter_daily} trades/day — forces you to wait for the highest-quality setups only.</li>
          <li><strong>Standard limit:</strong> {suggested_daily} trades/day — allows active trading while discouraging noise trades.</li>
        </ul>
        <br>Once you hit your daily limit, <strong>close your trading platform and log your session</strong>. 
        The best traders treat their daily limit as a hard stop, not a guideline.
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Recommended Daily Limit", f"{suggested_daily} trades")
        col_b.metric("Conservative Daily Limit", f"{tighter_daily} trades")

    # Stop-loss discipline
    with st.expander("🛑 Stop-Loss Discipline", expanded=la["flagged"]):
        if avg_win > 0 and avg_loss > 0:
            suggested_rr  = max(round(loss_ratio * 0.6, 1), 1.5)
            suggested_stop = round((avg_loss / avg_win) * 0.5 * 100, 1)
        else:
            suggested_rr   = 1.5
            suggested_stop = 1.0

        st.markdown(f"""
        <div class="insight-box">
        Your current avg loss is <strong>${avg_loss:.2f}</strong> vs avg win of <strong>${avg_win:.2f}</strong> 
        (ratio: <strong>{loss_ratio:.2f}×</strong>). 
        {'This exceeds the healthy 1.5× threshold — disciplined stop-losses are urgent.' if la["flagged"] else 'Maintain this by pre-setting stops on every trade.'}
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Target Risk/Reward",   f"1 : {suggested_rr}")
        r2.metric("Suggested Max Loss/Trade", f"${min(avg_loss * 0.6, avg_win):.2f}")
        r3.metric("Current Loss Ratio",   f"{loss_ratio:.2f}×")

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

    # Cooling-off periods
    with st.expander("❄️ Cooling-Off Periods", expanded=rt["flagged"]):
        st.markdown(f"""
        <div class="insight-box">
        {'<strong>⚠️ Revenge trading detected.</strong> You have opened oversized positions after losses ' + str(revenge_count) + ' time(s). A mandatory cooling-off protocol is strongly recommended.' if rt["flagged"] else '✅ No revenge trades detected — maintain this by keeping a cooling-off habit after any loss.'}
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown("""
        <div class="gauge-card">
            <div class="gauge-title">After a loss</div>
            <div class="gauge-val" style="color:#f59e0b;">30</div>
            <div class="gauge-badge" style="color:#f59e0b;">min cooldown</div>
        </div>
        """, unsafe_allow_html=True)
        c2.markdown("""
        <div class="gauge-card">
            <div class="gauge-title">After 3 losses in a row</div>
            <div class="gauge-val" style="color:#ef4444;">180</div>
            <div class="gauge-badge" style="color:#ef4444;">min cooldown</div>
        </div>
        """, unsafe_allow_html=True)
        c3.markdown("""
        <div class="gauge-card">
            <div class="gauge-title">After daily limit hit</div>
            <div class="gauge-val" style="color:#8b5cf6;">OFF</div>
            <div class="gauge-badge" style="color:#8b5cf6;">close platform</div>
        </div>
        """, unsafe_allow_html=True)

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

    # Overtrading control
    with st.expander("⏱️ Frequency Control", expanded=ot["flagged"]):
        best_hours = []
        if not hourly.empty:
            df_t2 = df.copy()
            df_t2["hour"] = df_t2["timestamp"].dt.hour
            hour_pl = df_t2.groupby("hour")["profit_loss"].mean()
            best_hours = hour_pl.nlargest(3).index.tolist()

        st.markdown(f"""
        <div class="insight-box">
        {'<strong>Your best-performing hours</strong> based on average P/L are: <strong>' + ', '.join([f"{h:02d}:00" for h in best_hours]) + '</strong>. Consider concentrating your trading in these windows.' if best_hours else 'Upload more data to see your best-performing hours.'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rec-card" style="border-color:#d97706;">
            <h4>⏱️ Frequency Control Rules</h4>
            <p>
            1. <strong>Define your trading window</strong> — pick 1–2 sessions per day (e.g. open + midday) and stay out otherwise.<br>
            2. Use a <strong>pre-trade checklist</strong>: setup present? Volume confirmed? News risk checked?<br>
            3. <strong>Quantity over quantity</strong> — five high-conviction trades beat twenty noise trades every time.<br>
            4. Track your win rate separately for planned vs impulse trades — the data will convince you faster than any rule.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — JOURNALING PROMPTS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">📓 Trading Psychology Journal Prompts</p>', unsafe_allow_html=True)
    st.markdown("Use these after each session. Write freely — there are no wrong answers. The goal is pattern recognition across weeks, not a single session.")

    prompt_tabs = st.tabs(["🔄 Overtrading", "😰 Loss Aversion", "😤 Revenge Trading", "🌅 Daily Reflection"])

    with prompt_tabs[0]:
        prompts_ot = [
            ("Before the session", "What was my plan going in? How many trades was I expecting to place, and why?"),
            ("After the session", "Did I take any trades where I couldn't immediately articulate a clear reason? What was I feeling when I entered?"),
            ("Pattern recognition", "What time of day did most of my trades cluster? Was I bored, anxious, or excited during those periods?"),
            ("FOMO check", "Which trades did I take because I was afraid of missing a move — not because my setup was present?"),
        ]
        for label, q in prompts_ot:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[1]:
        prompts_la = [
            ("On losing trades", "At what point did I first think about closing this trade? What stopped me from acting on that instinct?"),
            ("On winners exited early", "I closed this winner at {X}. What would have happened if I had stayed in to my original target?"),
            ("Anchoring check", "Am I holding because the trade has merit, or because I'm waiting to break even on my entry price?"),
            ("Scenario flip", "If I had no position right now, would I open this trade at the current price? If not — why am I holding it?"),
        ]
        for label, q in prompts_la:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[2]:
        prompts_rt = [
            ("After a loss", "Rate my emotional state after this loss: 1–10. At what number am I safe to trade again?"),
            ("Pre-trade check", "Am I placing this trade because the setup is valid, or because I lost money earlier and want to recover it?"),
            ("Size check", "Is this trade larger than my usual size? If yes — what is the reason? Is it based on a larger edge, or emotion?"),
            ("Consequence visualisation", "If this trade also loses, what will I feel? Am I comfortable with that outcome, or am I already in a reactive state?"),
        ]
        for label, q in prompts_rt:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    with prompt_tabs[3]:
        prompts_daily = [
            ("Session summary", "In one sentence: was today a process-driven day or an outcome-driven day? What's the difference in how I felt?"),
            ("Best decision", "What was the best decision I made today — not necessarily the most profitable, but the most disciplined?"),
            ("One thing to change", "If I could replay today, what is the single thing I would do differently? How will I make that change tomorrow?"),
            ("Gratitude & growth", "What did the market teach me today that I didn't know when I woke up?"),
        ]
        for label, q in prompts_daily:
            st.markdown(f'<div class="journal-card"><strong>{label}:</strong><br>"{q}"</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — INTERACTIVE JOURNAL ENTRY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<p class="section-head">✍️ Log a Journal Entry</p>', unsafe_allow_html=True)
    st.markdown("Record your reflections directly in the app. Entries are stored for the session.")

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    with st.form("journal_form"):
        j_date   = st.date_input("Session date", value=datetime.today())
        j_mood   = st.select_slider("Emotional state before session", options=["Very Calm", "Calm", "Neutral", "Anxious", "Very Anxious"], value="Neutral")
        j_plan   = st.text_area("What was your pre-session plan?", placeholder="E.g. Max 3 trades, only BTC, wait for support level confirmation...")
        j_debrief= st.text_area("Post-session debrief", placeholder="What happened? Did you stick to the plan?")
        j_bias   = st.multiselect("Which biases showed up today (self-assessed)?", ["Overtrading", "Loss Aversion", "Revenge Trading", "FOMO", "None"])
        j_lesson = st.text_area("One lesson from today", placeholder="e.g. I need to wait 10 min after entry before checking P/L...")
        submit   = st.form_submit_button("💾 Save Entry", type="primary")

    if submit:
        entry = {
            "date":     str(j_date),
            "mood":     j_mood,
            "plan":     j_plan,
            "debrief":  j_debrief,
            "biases":   ", ".join(j_bias) if j_bias else "None",
            "lesson":   j_lesson,
        }
        st.session_state.journal_entries.append(entry)
        st.success("✅ Journal entry saved!")

    if st.session_state.journal_entries:
        st.markdown("**Previous Entries This Session:**")
        st.dataframe(pd.DataFrame(st.session_state.journal_entries), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.caption("NBC Bias Detector · Feedback & Recommendations · Educational purposes only — not financial advice.")