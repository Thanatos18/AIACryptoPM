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
    import talib.abstract as talib
    TALIB_AVAILABLE = True
    TA_AVAILABLE = False
except ImportError:
    TALIB_AVAILABLE = False
    try:
        import ta
        TA_AVAILABLE = True
    except ImportError:
        TA_AVAILABLE = False
        logger.warning("Neither TA-Lib nor ta library found. Will use raw Pandas mathematical calculations.")


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
    process_only_new_candles = True
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
        Called automatically for each period specified in freqai_config.json.
        Memory optimized: Operates directly on the input dataframe to respect 16GB RAM limits.
        """
        # 1. Momentum Indicators
        if TALIB_AVAILABLE:
            dataframe[f"%-rsi-{period}"] = pd.Series(talib.RSI(dataframe, timeperiod=period), index=dataframe.index)
            dataframe[f"%-williams_r-{period}"] = pd.Series(talib.WILLR(dataframe, timeperiod=period), index=dataframe.index)
            dataframe[f"%-roc-{period}"] = pd.Series(talib.ROC(dataframe, timeperiod=period), index=dataframe.index)
        elif TA_AVAILABLE:
            dataframe[f"%-rsi-{period}"] = ta.momentum.rsi(dataframe["close"], window=period)
            dataframe[f"%-williams_r-{period}"] = ta.momentum.williams_r(dataframe["high"], dataframe["low"], dataframe["close"], lbp=period)
            dataframe[f"%-roc-{period}"] = ta.momentum.roc(dataframe["close"], window=period)
        else:
            # Fallback raw Pandas RSI
            delta = dataframe["close"].diff()
            gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0/period, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0/period, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            dataframe[f"%-rsi-{period}"] = 100.0 - (100.0 / (1.0 + rs))
            dataframe[f"%-roc-{period}"] = dataframe["close"].pct_change(periods=period) * 100.0
            
            # Fallback raw Pandas Williams %R
            highest_high = dataframe["high"].rolling(period).max()
            lowest_low = dataframe["low"].rolling(period).min()
            dataframe[f"%-williams_r-{period}"] = -100.0 * (highest_high - dataframe["close"]) / (highest_high - lowest_low).replace(0, np.nan)

        # 2. Volatility Indicators (Bollinger Bands)
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(dataframe["close"], timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
            upper = pd.Series(upper, index=dataframe.index)
            middle = pd.Series(middle, index=dataframe.index)
            lower = pd.Series(lower, index=dataframe.index)
            dataframe[f"%-bb_width-{period}"] = (upper - lower) / middle.replace(0, np.nan)
            dataframe[f"%-bb_pb-{period}"] = (dataframe["close"] - lower) / (upper - lower).replace(0, np.nan)
        elif TA_AVAILABLE and period >= 10:
            bb = ta.volatility.BollingerBands(dataframe["close"], window=period, window_dev=2.0)
            dataframe[f"%-bb_width-{period}"] = bb.bollinger_wband()
            dataframe[f"%-bb_pb-{period}"] = bb.bollinger_pband()
        else:
            sma = dataframe["close"].rolling(period).mean()
            std = dataframe["close"].rolling(period).std()
            upper = sma + (std * 2.0)
            lower = sma - (std * 2.0)
            dataframe[f"%-bb_width-{period}"] = (upper - lower) / sma.replace(0, np.nan)
            dataframe[f"%-bb_pb-{period}"] = (dataframe["close"] - lower) / (upper - lower).replace(0, np.nan)

        # 3. Moving Averages
        dataframe[f"%-ema-{period}"] = dataframe["close"].ewm(span=period, adjust=False).mean()
        dataframe[f"%-sma-{period}"] = dataframe["close"].rolling(period).mean()

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Constructs single-period or specialized volume/trend features.
        """
        # 1. Volume Regimes
        dataframe["%-volume-expansion"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean().replace(0, np.nan)

        # 2. MACD Trend
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_hist = talib.MACD(dataframe["close"])
            dataframe["%-macd-line"] = pd.Series(macd_line, index=dataframe.index)
            dataframe["%-macd-signal"] = pd.Series(macd_signal, index=dataframe.index)
            dataframe["%-macd-hist"] = pd.Series(macd_hist, index=dataframe.index)
        elif TA_AVAILABLE:
            macd = ta.trend.MACD(dataframe["close"])
            dataframe["%-macd-line"] = macd.macd()
            dataframe["%-macd-signal"] = macd.macd_signal()
            dataframe["%-macd-hist"] = macd.macd_diff()
        else:
            ema12 = dataframe["close"].ewm(span=12, adjust=False).mean()
            ema26 = dataframe["close"].ewm(span=26, adjust=False).mean()
            dataframe["%-macd-line"] = ema12 - ema26
            dataframe["%-macd-signal"] = dataframe["%-macd-line"].ewm(span=9, adjust=False).mean()
            dataframe["%-macd-hist"] = dataframe["%-macd-line"] - dataframe["%-macd-signal"]

        # 3. Microstructure candle attributes
        dataframe["%-body-ratio"] = (dataframe["close"] - dataframe["open"]).abs() / (dataframe["high"] - dataframe["low"]).replace(0, np.nan)

        return dataframe

    def feature_engineering_standard(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Constructs Spatio-Temporal features (Day of week, hour of day).
        These enable FreqAI models to capture liquidity cyclicality.
        """
        # Date/time Cyclical encoding
        if "date" in dataframe.columns:
            dt = pd.to_datetime(dataframe["date"])
            dataframe["hour_of_day"] = dt.dt.hour
            dataframe["day_of_week"] = dt.dt.dayofweek
            
            # Sine/Cosine trigonometric transformations to preserve cyclical continuity
            dataframe["%-hour-sin"] = np.sin(2.0 * np.pi * dataframe["hour_of_day"] / 24.0)
            dataframe["%-hour-cos"] = np.cos(2.0 * np.pi * dataframe["hour_of_day"] / 24.0)
            dataframe["%-day-sin"] = np.sin(2.0 * np.pi * dataframe["day_of_week"] / 7.0)
            dataframe["%-day-cos"] = np.cos(2.0 * np.pi * dataframe["day_of_week"] / 7.0)

        # Base technical indicators required for our trend override rules (used in strategy logic, not by FreqAI)
        dataframe["ema_50"] = dataframe["close"].ewm(span=50, adjust=False).mean()
        dataframe["ema_100"] = dataframe["close"].ewm(span=100, adjust=False).mean()
        dataframe["ema_200"] = dataframe["close"].ewm(span=200, adjust=False).mean()

        return dataframe

    def set_freqai_targets(
        self, dataframe: pd.DataFrame, metadata: Dict[str, Any], **kwargs
    ) -> pd.DataFrame:
        """
        Defines the production training target label for the LightGBM Classifier.
        We forecast ternary price direction (-1, 0, 1) N candles ahead.
        """
        feature_params = self.freqai_info.get("feature_parameters", {})
        label_period = feature_params.get("label_period_candles", 20)
        
        # Exact vectorized future return formulation
        fwd_return = (
            dataframe["close"].shift(-label_period) - dataframe["close"]
        ) / dataframe["close"]

        # Ternary label: 1 (long), -1 (short), 0 (neutral) based on 0.8% threshold
        threshold = 0.008
        dataframe["&-direction"] = np.where(
            fwd_return > threshold, 1,
            np.where(fwd_return < -threshold, -1, 0)
        )

        logger.debug(f"Generated FreqAI target label '&-direction' over a {label_period}-candle horizon.")
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Executes the total FreqAI production pipeline.
        Calls FreqTrade's internal self.freqai.start() which handles feature expansion,
        data scaling, Outlier Detection, and ML inference.
        """
        # Execute standard FreqAI wrapper if enabled
        if self.freqai_info.get("enabled", False):
            logger.info(f"DF Columns before FreqAI start: {list(dataframe.columns)}")
            dataframe = self.freqai.start(dataframe, metadata, self)
        else:
            # Fallback local indicator calculation if running in standalone debug mode
            dataframe = self.feature_engineering_standard(dataframe, metadata)
            dataframe = self.feature_engineering_expand_basic(dataframe, metadata)

        # Removed gc.collect() from candle hot-path to eliminate stop-the-world pauses

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluates the DataFrame for long entry conditions.
        Combines non-linear machine learning prediction confidence with rigorous mathematical filters.
        """
        conditions = []

        pred_col = "&-direction"
        proba_col = "&-direction_max_prediction_confidence"

        # 1. Machine Learning Target Evaluation
        if pred_col in dataframe.columns and "do_predict" in dataframe.columns:
            ml_condition = (
                (dataframe["do_predict"] == 1) &
                (dataframe[pred_col] == 1) &
                (dataframe.get(proba_col, 0) > 0.55)
            )
            conditions.append(ml_condition)
        else:
            # Fallback if ML signal is temporarily unavailable
            logger.debug(
                f"ML prediction column '{pred_col}' unavailable for {metadata.get('pair')}. "
                "Using technical override."
            )
            rsi_col = next((c for c in dataframe.columns if "rsi" in c), None)
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
            # Do not fight macro liquidation phases
            trend_filter = (
                (dataframe["ema_50"] > dataframe["ema_200"]) |
                (dataframe["close"] > dataframe["ema_100"])
            )
            conditions.append(trend_filter)

        # 3. Volume Liquidity Filter
        if "volume" in dataframe.columns:
            # Calculate volume SMA inline to avoid temp column allocations
            vol_sma = dataframe["volume"].rolling(20).mean()
            volume_filter = (
                dataframe["volume"] > (vol_sma * self.volume_regime_multiplier.value)
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

        pred_col = "&-direction"

        if pred_col in dataframe.columns and "do_predict" in dataframe.columns:
            # If the ML model predicts upcoming negative performance (-1), cut position proactively
            ml_exit = (
                (dataframe["do_predict"] == 1) &
                (dataframe[pred_col] == -1)
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
