#!/usr/bin/env python3
"""
Production CPU Analytics Dashboard for FreqTrade + FreqAI
Author: Senior Quantitative Developer & Algorithmic Trading Systems Architect

This Streamlit dashboard performs advanced diagnostics on your FreqTrade SQLite
database ('tradesv3.sqlite') and visualizes machine learning feature importances.
Designed to execute smoothly on local Windows laptops without spiking RAM usage.
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Parse CLI args for database file selection
default_db = "tradesv3.demo.sqlite" if "--demo" in sys.argv else "tradesv3.sqlite"

# Streamlit Page Setup
st.set_page_config(
    page_title="FreqTrade + FreqAI Quant Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium quantitative look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .metric-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stProgress > div > div > div > div { background-color: #00f2fe; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_trades_data(db_path: str = "tradesv3.sqlite", limit_30_days: bool = True) -> pd.DataFrame:
    """Safely loads closed trade history from FreqTrade SQLite database."""
    path = Path(db_path)
    if not path.exists():
        # Fallback check if script is executed from top level directory
        path = Path("production_crypto_freqai") / db_path
        if not path.exists():
            return pd.DataFrame()

    try:
        conn = sqlite3.connect(path)
        
        # Build query: limit to last 30 days if limit_30_days is True
        where_clause = ""
        if limit_30_days:
            where_clause = "WHERE (close_date IS NULL) OR (close_date >= datetime('now', '-30 days'))"
            
        query = f"""
            SELECT id, exchange, pair, is_open, open_rate, close_rate,
                   close_profit, close_profit_abs, stake_amount,
                   open_date, close_date, exit_reason, is_short
            FROM trades
            {where_clause}
            ORDER BY open_date ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df["open_date"] = pd.to_datetime(df["open_date"])
            df["close_date"] = pd.to_datetime(df["close_date"])
            df["trade_duration_min"] = (df["close_date"] - df["open_date"]).dt.total_seconds() / 60.0
            
            # Downcast to float32 to conserve memory
            for col in ["open_rate", "close_rate", "close_profit", "close_profit_abs", "stake_amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], downcast="float")
            
        return df
    except Exception as e:
        st.error(f"Error loading SQLite database: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_feature_importances() -> Dict[str, Dict[str, float]]:
    """Loads FreqAI LightGBM/CatBoost serialized feature importance files recursively."""
    models_dir = Path("user_data/models")
    if not models_dir.exists():
        models_dir = Path("production_crypto_freqai/user_data/models")
        if not models_dir.exists():
            return {}

    importance_data = {}
    # Glob both LightGBM and CatBoost files recursively
    patterns = ["**/feature_importance_*.json", "**/catboost_feature_importance_*.json"]
    
    for pattern in patterns:
        for json_file in models_dir.glob(pattern):
            stem = json_file.stem
            
            # Remove prefix
            if stem.startswith("catboost_feature_importance_"):
                clean_stem = stem.replace("catboost_feature_importance_", "")
            else:
                clean_stem = stem.replace("feature_importance_", "")
            
            # Parse timestamp if present (e.g. BTC_USDT_1781409747 -> BTC_USDT)
            parts = clean_stem.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                parts = parts[:-1]
            
            pair_name = "/".join(parts) # Map BTC_USDT back to BTC/USDT
            if not pair_name:
                continue

            try:
                mtime = json_file.stat().st_mtime
                # Only keep the most recent file per pair
                if pair_name not in importance_data or mtime > importance_data[pair_name]["mtime"]:
                    with open(json_file, "r") as f:
                        importance_data[pair_name] = {
                            "data": json.load(f),
                            "mtime": mtime
                        }
            except Exception:
                continue
                
    return {k: v["data"] for k, v in importance_data.items()}


def compute_drawdown(equity_series: pd.Series) -> Tuple[pd.Series, float]:
    """Computes running drawdown series and maximum drawdown percentage."""
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min() * 100.0  # Min value is the deepest drawdown
    return drawdown * 100.0, max_dd


def render_dashboard() -> None:
    """Renders the entire multi-tab quantitative Streamlit interface."""
    st.title("⚡ FreqTrade + FreqAI Production Analytics Dashboard")
    st.markdown("Real-time telemetry, quantitative portfolio tracking, and machine learning model explainability.")

    # Sidebar settings
    st.sidebar.title("Telemetry Settings")
    show_full_history = st.sidebar.checkbox(
        "Show full trade history", 
        value=False, 
        help="If unchecked, loads only the last 30 days of closed trades to conserve memory."
    )

    # Load Telemetry
    df = load_trades_data(default_db, limit_30_days=not show_full_history)

    if df.empty:
        st.warning(f"⚠️ No trades found in '{default_db}'. Please execute `python scripts/simulate_trades_db.py` to compile demo executions.")
        return

    # Split open vs closed trades
    closed_trades = df[df["is_open"] == 0].copy()
    open_trades = df[df["is_open"] == 1].copy()

    # Initial Wallet Base
    initial_wallet = 10000.0

    if not closed_trades.empty:
        closed_trades["running_pnl"] = closed_trades["close_profit_abs"].cumsum()
        closed_trades["equity"] = initial_wallet + closed_trades["running_pnl"]
        
        total_pnl_abs = closed_trades["close_profit_abs"].sum()
        total_pnl_pct = (total_pnl_abs / initial_wallet) * 100.0
        
        wins = closed_trades[closed_trades["close_profit"] > 0]
        losses = closed_trades[closed_trades["close_profit"] < 0]
        win_rate = (len(wins) / len(closed_trades)) * 100.0 if len(closed_trades) > 0 else 0.0
        
        profit_factor = wins["close_profit_abs"].sum() / abs(losses["close_profit_abs"].sum()) if abs(losses["close_profit_abs"].sum()) > 0 else 99.0
        
        dd_series, max_dd = compute_drawdown(closed_trades["equity"])
        closed_trades["drawdown"] = dd_series
        
        # Approximate Annualized Sharpe Ratio (assuming ~15 trades / week)
        returns = closed_trades["close_profit"]
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(730) if returns.std() > 0 else 0.0
        calmar_ratio = (total_pnl_pct / abs(max_dd)) if abs(max_dd) > 0 else 0.0

        # Top Level Metric Cards
        cols = st.columns(6)
        with cols[0]:
            st.metric("Net Capital", f"${closed_trades['equity'].iloc[-1]:,.2f}", f"{total_pnl_pct:+.2f}%")
        with cols[1]:
            st.metric("Total PnL (USDT)", f"${total_pnl_abs:+,.2f}", f"{len(closed_trades)} Trades")
        with cols[2]:
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{len(wins)}W / {len(losses)}L")
        with cols[3]:
            st.metric("Max Drawdown", f"{max_dd:.2f}%", f"Calmar: {calmar_ratio:.2f}")
        with cols[4]:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        with cols[5]:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "OOS Benchmark")

        st.divider()

        # Multi-Tab Layout
        tab_equity, tab_pairs, tab_accuracy, tab_freqai, tab_open = st.tabs([
            "📈 Equity & Drawdown", 
            "📊 Asset Performance", 
            "🎯 Prediction Accuracy", 
            "🧠 FreqAI ML Feature Explainer", 
            "🟢 Open Positions"
        ])

        # TAB 1: Equity & Drawdown Curves
        with tab_equity:
            fig_equity = px.line(
                closed_trades, x="close_date", y="equity", 
                title="Cumulative Portfolio Equity Curve (USDT)",
                labels={"close_date": "Date", "equity": "Total Balance (USDT)"},
                color_discrete_sequence=["#00f2fe"]
            )
            fig_equity.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_equity, use_container_width=True)

            fig_dd = px.area(
                closed_trades, x="close_date", y="drawdown", 
                title="Portfolio Max Drawdown Tracking (%)",
                labels={"close_date": "Date", "drawdown": "Drawdown (%)"},
                color_discrete_sequence=["#ff4b4b"]
            )
            fig_dd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_dd, use_container_width=True)

        # TAB 2: Asset Performance
        with tab_pairs:
            pair_summary = closed_trades.groupby("pair").agg(
                trades=("id", "count"),
                total_pnl=("close_profit_abs", "sum"),
                win_rate=("close_profit", lambda x: (x > 0).mean() * 100.0)
            ).reset_index()

            fig_bar = px.bar(
                pair_summary, x="pair", y="total_pnl", color="win_rate",
                title="Total Net Profit by Cryptocurrency Pair (USDT)",
                labels={"pair": "Trading Pair", "total_pnl": "Net Profit (USDT)", "win_rate": "Win Rate (%)"},
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### Detailed Cryptocurrency Pair Breakdown")
            st.dataframe(pair_summary.style.format({
                "total_pnl": "${:,.2f}",
                "win_rate": "{:.1f}%"
            }), use_container_width=True)

        # TAB 3: Prediction Accuracy over time
        with tab_accuracy:
            st.markdown("### Rolling 20-Trade Win Rate & Signal Consistency")
            closed_trades["rolling_winrate"] = closed_trades["close_profit"].rolling(20).apply(lambda x: (x > 0).mean() * 100.0)
            
            fig_acc = px.line(
                closed_trades.dropna(subset=["rolling_winrate"]), x="close_date", y="rolling_winrate",
                title="Rolling 20-Trade Prediction Win Rate (%)",
                labels={"close_date": "Date", "rolling_winrate": "Rolling Win Rate (%)"},
                color_discrete_sequence=["#00b09b"]
            )
            fig_acc.add_hline(y=60.0, line_dash="dash", line_color="yellow", annotation_text="Minimum Threshold Target (60%)")
            fig_acc.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_acc, use_container_width=True)

            # Exit reasons pie chart
            fig_pie = px.pie(
                closed_trades, names="exit_reason", title="Trade Exit Regimes Summary",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        # TAB 4: FreqAI ML Explainer
        with tab_freqai:
            st.markdown("### LightGBM / CatBoost Top Feature Importance Tracking")
            st.markdown("This visualizer demonstrates how the rolling multi-day ML model weights different technical indicator dimensions in real time.")

            feat_importances = load_feature_importances()

            if not feat_importances:
                st.info("⚠️ Feature importance tracking JSON files not yet generated. They will appear here once the first FreqAI model retraining cycle finishes.")
            else:
                pair_select = st.selectbox("Select Model Trading Pair / Scope", list(feat_importances.keys()))
                
                feat_dict = feat_importances[pair_select]
                feat_df = pd.DataFrame(list(feat_dict.items()), columns=["Feature", "Importance Score"])
                feat_df = feat_df.sort_values(by="Importance Score", ascending=False).head(15)

                fig_feat = px.bar(
                    feat_df, x="Importance Score", y="Feature", orientation="h",
                    title=f"Top 15 Most Impactful Technical Features -> {pair_select}",
                    color="Importance Score", color_continuous_scale="Plasma"
                )
                fig_feat.update_layout(yaxis={"categoryorder": "total ascending"}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_feat, use_container_width=True)

        # TAB 5: Open Positions
        with tab_open:
            if open_trades.empty:
                st.success("🟢 No active trades open right now. Waiting for upcoming FreqAI ML prediction confidence thresholds to trigger.")
            else:
                st.markdown("### Currently Active Live Market Executions")
                st.dataframe(open_trades[["id", "exchange", "pair", "open_rate", "stake_amount", "open_date"]].style.format({
                    "open_rate": "${:,.4f}",
                    "stake_amount": "${:,.2f}"
                }), use_container_width=True)

    else:
        st.info("ℹ️ No closed trades available yet. System is active and logging real-time data.")


if __name__ == "__main__":
    render_dashboard()
