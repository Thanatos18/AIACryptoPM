#!/usr/bin/env python3
"""
Production Helper Script: FreqTrade Database & Feature Importance Simulator
Author: Senior Quantitative Developer & Trading Systems Architect

This script populates a sample SQLite database ('tradesv3.sqlite') and generates
simulated LightGBM/CatBoost feature importance tracking files.
It allows the user to immediately experience and test the supplementary
Streamlit / Plotly analytics dashboard without waiting weeks for live trade data.
"""

import json
import logging
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SimulationBuilder")


def simulate_feature_importances() -> None:
    """Generates realistic machine learning feature importance JSON files."""
    models_dir = Path("user_data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    pairs = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "LINK_USDT", "AVAX_USDT", "general"]

    # Base realistic features derived from our Strategy Research Report
    base_features = {
        "&-price_return_pred": 0.284,
        "rsi_14": 0.152,
        "bb_width_21": 0.118,
        "volume_expansion": 0.095,
        "macd_hist": 0.082,
        "body_to_range": 0.064,
        "bb_pb_21": 0.051,
        "ema_50": 0.045,
        "williams_r_14": 0.038,
        "roc_14": 0.031,
        "hour_sin": 0.022,
        "day_cos": 0.018
    }

    for pair in pairs:
        # Add slight statistical noise to make each pair unique
        noisy_feats = {}
        for feat, imp in base_features.items():
            noise = random.uniform(0.85, 1.15)
            noisy_feats[feat] = round(imp * noise, 4)

        # Re-sort descending
        sorted_feats = dict(sorted(noisy_feats.items(), key=lambda item: item[1], reverse=True))

        dump_path = models_dir / f"feature_importance_{pair}.json"
        with open(dump_path, "w") as f:
            json.dump(sorted_feats, f, indent=4)
        
        logger.info(f"Generated sample ML feature importance file -> {dump_path}")


def simulate_trades_database() -> None:
    """Constructs a fully functional FreqTrade SQLite database (tradesv3.demo.sqlite)."""
    db_path = Path("tradesv3.demo.sqlite")
    
    # Remove existing demo DB if present
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Replicate FreqTrade trades schema exactly
    cursor.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange VARCHAR NOT NULL,
            pair VARCHAR NOT NULL,
            is_open BOOLEAN NOT NULL,
            fee_open_cost FLOAT NOT NULL,
            fee_open_currency VARCHAR NOT NULL,
            fee_close_cost FLOAT,
            fee_close_currency VARCHAR,
            open_rate FLOAT NOT NULL,
            close_rate FLOAT,
            close_profit FLOAT,
            close_profit_abs FLOAT,
            stake_amount FLOAT NOT NULL,
            amount FLOAT NOT NULL,
            open_date DATETIME NOT NULL,
            close_date DATETIME,
            open_order_id VARCHAR,
            stop_loss FLOAT,
            stop_loss_pct FLOAT,
            initial_stop_loss FLOAT,
            initial_stop_loss_pct FLOAT,
            stoploss_order_id VARCHAR,
            stoploss_last_update DATETIME,
            max_rate FLOAT,
            min_rate FLOAT,
            exit_reason VARCHAR,
            exit_order_status VARCHAR,
            strategy VARCHAR,
            enter_tag VARCHAR,
            timeframe INTEGER,
            trading_mode VARCHAR,
            leverage FLOAT NOT NULL,
            is_short BOOLEAN NOT NULL,
            funding_fees FLOAT,
            liquidation_price FLOAT
        )
    """)

    logger.info("Created SQLite 'trades' schema successfully.")

    # Generate 150 simulated trades over the past 90 days
    pairs_data = {
        "BTC/USDT": {"base_price": 64000.0, "volatility": 0.02},
        "ETH/USDT": {"base_price": 35000.0, "volatility": 0.035},  # Scaled dummy base
        "SOL/USDT": {"base_price": 140.0, "volatility": 0.06},
        "LINK/USDT": {"base_price": 18.0, "volatility": 0.05},
        "AVAX/USDT": {"base_price": 32.0, "volatility": 0.055},
        "BNB/USDT": {"base_price": 610.0, "volatility": 0.03}
    }

    exit_reasons = [
        ("roi", 0.45, 0.015, 0.055),           # Reason, Probability, min profit, max profit
        ("trailing_stop", 0.30, 0.005, 0.040),
        ("exit_signal", 0.15, -0.015, 0.025),
        ("stop_loss", 0.10, -0.060, -0.045)
    ]

    current_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()

    trade_id = 1
    total_capital = 10000.0

    while current_date < end_date:
        pair = random.choice(list(pairs_data.keys()))
        pdata = pairs_data[pair]

        # Simulate price drifting
        pdata["base_price"] *= random.uniform(0.99, 1.01)
        open_price = pdata["base_price"]
        
        stake = total_capital * 0.15  # 15% position sizing
        amount = stake / open_price

        # Pick exit reason based on distribution
        rand_val = random.random()
        cum_prob = 0.0
        selected_exit = exit_reasons[0]
        for er in exit_reasons:
            cum_prob += er[1]
            if rand_val <= cum_prob:
                selected_exit = er
                break

        profit_pct = random.uniform(selected_exit[2], selected_exit[3])
        close_price = open_price * (1.0 + profit_pct)
        profit_abs = stake * profit_pct

        total_capital += profit_abs

        open_dt = current_date
        hold_minutes = random.randint(15, 360)
        close_dt = open_dt + timedelta(minutes=hold_minutes)

        cursor.execute("""
            INSERT INTO trades (
                id, exchange, pair, is_open, fee_open_cost, fee_open_currency,
                fee_close_cost, fee_close_currency, open_rate, close_rate,
                close_profit, close_profit_abs, stake_amount, amount,
                open_date, close_date, exit_reason, strategy, timeframe,
                trading_mode, leverage, is_short
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, "binance", pair, False, stake * 0.001, "USDT",
            (stake + profit_abs) * 0.001, "USDT", open_price, close_price,
            profit_pct, profit_abs, stake, amount,
            open_dt.strftime("%Y-%m-%d %H:%M:%S.%f"), close_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            selected_exit[0], "FreqAiAdaptiveRollingStrategy", 15,
            "spot", 1.0, False
        ))

        trade_id += 1
        # Advance time between 1 and 18 hours
        current_date += timedelta(hours=random.uniform(1.0, 18.0))

    # Add 2 currently open trades
    for _ in range(2):
        pair = random.choice(list(pairs_data.keys()))
        open_price = pairs_data[pair]["base_price"]
        stake = 1500.0
        amount = stake / open_price
        open_dt = datetime.now() - timedelta(minutes=random.randint(10, 60))

        cursor.execute("""
            INSERT INTO trades (
                id, exchange, pair, is_open, fee_open_cost, fee_open_currency,
                open_rate, stake_amount, amount, open_date, strategy, timeframe,
                trading_mode, leverage, is_short
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, "binance", pair, True, stake * 0.001, "USDT",
            open_price, stake, amount, open_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "FreqAiAdaptiveRollingStrategy", 15, "spot", 1.0, False
        ))
        trade_id += 1

    conn.commit()
    conn.close()
    
    logger.info(f"Successfully populated 'tradesv3.demo.sqlite' with {trade_id - 1} simulated executions.")
    logger.info(f"Simulated Ending Capital: ~{total_capital:.2f} USDT")


if __name__ == "__main__":
    simulate_feature_importances()
    simulate_trades_database()
    print("\n[SUCCESS] Simulation DB successfully compiled. You can now execute:")
    print("    streamlit run scripts/cpu_analytics_dashboard.py -- --demo")
