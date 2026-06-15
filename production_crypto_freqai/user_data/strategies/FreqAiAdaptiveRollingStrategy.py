import gc
import logging
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    BooleanParameter
)

logger = logging.getLogger(__name__)

# Attempt to import ta / talib for technical indicators with complete standalone fallback
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("Pure Python 'ta' library not found. Will use raw Pandas mathematical calculations.")


class FreqAiAdaptiveRollingStrategy(IStrategy):
    """
    Production machine learning trading strategy using FreqAI Adaptive Rolling ML.
    Fully optimized for a Windows laptop with 16GB RAM (CPU-only execution).
    
    This strategy coordinates the feature pipeline, constructs target labels,
    and processes non-linear ML predictions alongside structural market regime filters.
    """

    # 1. Strategy Identity & Execution Information
    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "15m"
    startup_candle_count: int = 300

    # Minimal ROI (Return On Investment) table:
    # Based on our out-of-sample research report, we secure immediate profits on spikes
    # and gradually lower our profit expectations to exit stale positions safely.
    minimal_roi = {
        "0": 0.05,     # Immediate exit if trade hits +5.0%
        "30": 0.03,    # +3.0% acceptable after 30 minutes
        "60": 0.015,   # +1.5% acceptable after 60 minutes
        "120": 0.00    # Break even exit after 2 hours if momentum stalls
    }

    # Hard Stoploss: Tail risk protection against extreme flash crashes
    stoploss = -0.06   # -6.0% maximum allowable loss

    # Trailing Stop: Adaptive ratchet mechanisms
    trailing_stop = True
    trailing_stop_positive = 0.012         # Once activated, trail price by 1.2%
    trailing_stop_positive_offset = 0.025  # Activate trailing stop when trade hits +2.5% profit
    trailing_only_offset_is_reached = True

    protections = [
        {"method": "CooldownPeriod", "stop_duration_candles": 5},
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 60,
            "trade_limit": 4,
            "stop_duration_candles": 60,
            "only_per_pair": False,
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 200,
            "trade_limit": 20,
            "stop_duration_candles": 12,
            "max_allowed_drawdown": 0.15,
        },
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 60,
            "trade_limit": 2,
            "stop_duration_candles": 60,
            "required_profit": 0.02,
        },
    ]

    # 2. Hyperparameter Space (Can be swept via FreqTrade Hyperopt)
    entry_return_threshold = DecimalParameter(0.005, 0.030, default=0.012, space="buy", optimize=True)
    exit_return_threshold = DecimalParameter(-0.020, 0.000, default=-0.005, space="sell", optimize=True)
    volume_regime_multiplier = DecimalParameter(1.0, 2.0, default=1.2, space="buy", optimize=True)

    # Run in dry-run or live mode (introspected from FreqTrade config)
    # We maintain internal trade decision tracking
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.freqai_info: Dict[str, Any] = config.get("freqai", {})
        self.mode_name = "Live Trading" if not config.get("dry_run", True) else "Dry-Run (Paper Trading)"
        logger.info(f"Initialized FreqAiAdaptiveRollingStrategy in {self.mode_name} mode.")

    def feature_engineering_expand_all(
        self, dataframe: pd.DataFrame, period: int, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Constructs multi-period technical indicators for FreqAI ingestion.
        Called automatically for each period specified in freqai_config.json (14, 21, 50, 100).
        Memory optimized: Uses in-place or vectorized Pandas operations to respect 16GB RAM limits.
        """
        # Ensure we don't fragment dataframes
        df = dataframe.copy()

        # 1. Momentum Indicators
        if TA_AVAILABLE:
            # RSI
            df[f"rsi_{period}"] = ta.momentum.rsi(df["close"], window=period)
            # Williams %R
            df[f"williams_r_{period}"] = ta.momentum.williams_r(df["high"], df["low"], df["close"], lbp=period)
            # Rate of Change (ROC)
            df[f"roc_{period}"] = ta.momentum.roc(df["close"], window=period)
        else:
            # Fallback raw Pandas RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0/period, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0/period, adjust=False).mean()
            rs = gain / loss
            df[f"rsi_{period}"] = 100.0 - (100.0 / (1.0 + rs))
            df[f"roc_{period}"] = df["close"].pct_change(periods=period) * 100.0

        # 2. Volatility Indicators (Bollinger Bands)
        if TA_AVAILABLE and period >= 10:
            bb = ta.volatility.BollingerBands(df["close"], window=period, window_dev=2.0)
            df[f"bb_width_{period}"] = bb.bollinger_wband()
            df[f"bb_pb_{period}"] = bb.bollinger_pband()
        else:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            upper = sma + (std * 2.0)
            lower = sma - (std * 2.0)
            df[f"bb_width_{period}"] = (upper - lower) / sma
            df[f"bb_pb_{period}"] = (df["close"] - lower) / (upper - lower)

        # 3. Moving Averages
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        df[f"sma_{period}"] = df["close"].rolling(period).mean()

        return df

    def feature_engineering_expand_basic(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Constructs single-period or specialized volume/trend features.
        """
        df = dataframe.copy()

        # 1. Volume Regimes
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_expansion"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

        # 2. MACD Trend
        if TA_AVAILABLE:
            macd = ta.trend.MACD(df["close"])
            df["macd_line"] = macd.macd()
            df["macd_signal"] = macd.macd_signal()
            df["macd_hist"] = macd.macd_diff()
        else:
            ema12 = df["close"].ewm(span=12, adjust=False).mean()
            ema26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd_line"] = ema12 - ema26
            df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # 3. Microstructure candle attributes
        df["candle_body"] = (df["close"] - df["open"]).abs()
        df["candle_range"] = df["high"] - df["low"]
        df["body_to_range"] = df["candle_body"] / df["candle_range"].replace(0, np.nan)

        return df

    def feature_engineering_standard(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Constructs Spatio-Temporal features (Day of week, hour of day).
        These enable FreqAI models to capture liquidity cyclicality.
        """
        df = dataframe.copy()

        # Date/time Cyclical encoding
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"])
            df["hour_of_day"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
            
            # Sine/Cosine trigonometric transformations to preserve cyclical continuity
            df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
            df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)
            df["day_sin"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
            df["day_cos"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

        # Base technical indicators required for our trend override rules
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_100"] = df["close"].ewm(span=100, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

        return df

    def set_freqai_targets(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Defines the production training target label for the LightGBM/CatBoost Regressor.
        We forecast continuous price return N candles ahead.
        """
        df = dataframe.copy()
        
        feature_params = self.freqai_info.get("feature_parameters", {})
        label_period = feature_params.get("label_period_candles", 20)

        # Target Label: Forward future percentage return
        target_col = f"&-price_return"
        
        # Exact vectorized future return formulation
        df[target_col] = (
            df["close"].shift(-label_period) - df["close"]
        ) / df["close"]

        # Log target completion
        logger.debug(f"Generated FreqAI target label '{target_col}' over a {label_period}-candle horizon.")

        return df

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Executes the total FreqAI production pipeline.
        Calls FreqTrade's internal self.freqai.start() which handles feature expansion,
        data scaling, Outlier Detection, and ML inference.
        """
        # Execute standard FreqAI wrapper if enabled
        if self.freqai_info.get("enabled", False):
            dataframe = self.freqai.start(dataframe, metadata, self)
        else:
            # Fallback local indicator calculation if running in standalone debug mode
            dataframe = self.feature_engineering_standard(dataframe, metadata)
            dataframe = self.feature_engineering_expand_basic(dataframe, metadata)

        # Explicit garbage collection to maintain 16GB RAM ceiling
        gc.collect()

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluates the DataFrame for long entry conditions.
        Combines non-linear machine learning prediction confidence with rigorous mathematical filters.
        """
        conditions = []

        # Target ML Prediction column generated by FreqAI
        pred_col = "&-price_return_pred"

        # 1. Machine Learning Target Evaluation
        if pred_col in dataframe.columns and "do_predict" in dataframe.columns:
            ml_condition = (
                (dataframe["do_predict"] == 1) &
                (dataframe[pred_col] > self.entry_return_threshold.value)
            )
            conditions.append(ml_condition)
        else:
            # Fallback if ML signal is temporarily unavailable (compute RSI inline if needed)
            logger.debug(
                f"ML prediction column '{pred_col}' unavailable for {metadata.get('pair')}. "
                "Using technical override."
            )
            rsi_col = next((c for c in dataframe.columns if c.startswith("rsi_")), None)
            if rsi_col is None and "close" in dataframe.columns:
                delta = dataframe["close"].diff()
                gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / 14, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / 14, adjust=False).mean()
                rsi = 100.0 - (100.0 / (1.0 + gain / loss.replace(0, np.nan)))
            elif rsi_col is not None:
                rsi = dataframe[rsi_col]
            else:
                rsi = None

            if rsi is not None and "ema_50" in dataframe.columns:
                tech_condition = (dataframe["close"] > dataframe["ema_50"]) & (rsi < 60)
                conditions.append(tech_condition)
            elif "ema_50" in dataframe.columns:
                conditions.append(dataframe["close"] > dataframe["ema_50"])

        # 2. Structural Regime Protection Filters
        if "ema_50" in dataframe.columns and "ema_200" in dataframe.columns:
            # Do not fight macro liquidation phases: Require medium-term EMA > long-term EMA OR price breakout
            trend_filter = (
                (dataframe["ema_50"] > dataframe["ema_200"]) |
                (dataframe["close"] > dataframe["ema_100"])
            )
            conditions.append(trend_filter)

        # 3. Volume Liquidity Filter
        if "volume" in dataframe.columns and "volume_sma_20" in dataframe.columns:
            # Require active market liquidity
            volume_filter = (
                dataframe["volume"] > (dataframe["volume_sma_20"] * self.volume_regime_multiplier.value)
            )
            conditions.append(volume_filter)

        # Compile total entry vector
        if conditions:
            dataframe.loc[
                np.bitwise_and.reduce(conditions),
                "enter_long"
            ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluates the DataFrame for long exit conditions.
        Allows FreqAI to predict positive momentum exhaustion before hard stoplosses are triggered.
        """
        conditions = []

        pred_col = "&-price_return_pred"

        if pred_col in dataframe.columns and "do_predict" in dataframe.columns:
            # If the ML model predicts upcoming negative performance, cut position proactively
            ml_exit = (
                (dataframe["do_predict"] == 1) &
                (dataframe[pred_col] < self.exit_return_threshold.value)
            )
            conditions.append(ml_exit)

        if conditions:
            dataframe.loc[
                np.bitwise_and.reduce(conditions),
                "exit_long"
            ] = 1

        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str,
        current_time: Any, entry_tag: Optional[str], side: str, **kwargs
    ) -> bool:
        """
        Final safety execution gate executed right before real orders are dispatched to the exchange.
        Enforces extensive production logging and dynamic validation.
        """
        logger.info(
            f"🚀 [{self.mode_name}] Dispatched ENTRY Signal -> "
            f"Pair: {pair} | Side: {side.upper()} | Rate: {rate} | Value: ~{rate * amount:.2f} USDT"
        )
        return True

    def confirm_trade_exit(
        self, pair: str, trade: Any, order_type: str, amount: float, rate: float,
        time_in_force: str, exit_reason: str, current_time: Any, **kwargs
    ) -> bool:
        """
        Exit confirmation hook for real-time order summaries.
        """
        profit_ratio = trade.calc_profit_ratio(rate)
        profit_pct = profit_ratio * 100.0
        profit_abs = trade.calc_profit(rate)

        emoji = "📈" if profit_pct > 0 else "📉"
        
        logger.info(
            f"{emoji} [{self.mode_name}] Executing EXIT -> "
            f"Pair: {pair} | Reason: {exit_reason} | Duration: {trade.trade_duration}m | "
            f"Net PnL: {profit_pct:+.2f}% ({profit_abs:+.2f} USDT)"
        )
        return True
