@echo off
SETLOCAL EnableDelayedExpansion
chcp 65001 >nul 2>&1

SET OMP_NUM_THREADS=4
SET OPENBLAS_NUM_THREADS=4
SET MKL_NUM_THREADS=4
SET NUMEXPR_NUM_THREADS=4

TITLE FreqTrade One-Click Dry-Run (FreqAI ML)

:: One-click dry-run launcher:
::   1. Turn on WARP (or ensure Binance is reachable)
::   2. Double-click this file
::   3. Bot downloads data, trains FreqAI models, and starts paper trading

cd /d "%~dp0"

ECHO =====================================================================
ECHO  FreqTrade + FreqAI -- ONE-CLICK DRY-RUN
ECHO  Simulated wallet: $10,000 USDT -- Timeframe: 15m
ECHO  Pairs: BTC, ETH, SOL, LINK, AVAX, BNB -- Model: LightGBMClassifierCPU
ECHO =====================================================================
ECHO.
ECHO  Step 1: Enable WARP (or your VPN) so Binance data can be fetched.
ECHO  Step 2: This script will download data, then start paper trading.
ECHO.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    ECHO [!] Virtual environment not found. Run setup first:
    ECHO     python -m venv .venv
    ECHO     .venv\Scripts\pip install -r requirements.txt
    PAUSE
    EXIT /B 1
)

ECHO [i] Activating Python virtual environment...
CALL .venv\Scripts\activate.bat

IF NOT EXIST "user_data\logs" MKDIR "user_data\logs"
IF NOT EXIST "user_data\models" MKDIR "user_data\models"
IF NOT EXIST "user_data\data" MKDIR "user_data\data"
IF NOT EXIST "user_data\freqai_models" MKDIR "user_data\freqai_models"

SET PYTHONIOENCODING=utf-8
SET PYTHONUTF8=1

SET CONFIG_ARGS=-c config.json -c freqai_config.json --user-data-dir user_data
IF EXIST config.local.json (
    SET CONFIG_ARGS=!CONFIG_ARGS! -c config.local.json
)
SET TRADE_ARGS=--strategy FreqAiAdaptiveRollingStrategy --strategy-path user_data\strategies --freqaimodel LightGBMClassifierCPU --freqaimodel-path user_data\freqai_models

ECHO.
ECHO [i] Downloading / updating market data (60 days, 15m + 1h)...
ECHO [i] If this step hangs, confirm WARP is connected.
ECHO.

python -m freqtrade download-data %CONFIG_ARGS% --exchange binance --pairs BTC/USDT ETH/USDT SOL/USDT LINK/USDT AVAX/USDT BNB/USDT --timeframes 15m 1h --days 60 --prepend

IF ERRORLEVEL 1 (
    ECHO.
    ECHO [!] Data download failed. Check WARP/network, then re-run this script.
    ECHO [!] Attempting to start bot with any existing local data...
    ECHO.
    TIMEOUT /T 5 /NOBREAK >nul
)

:: Start Streamlit Analytics Dashboard in a separate window
START "Streamlit Analytics Dashboard" cmd /k "CALL .venv\Scripts\activate.bat && python -m streamlit run scripts/cpu_analytics_dashboard.py"

ECHO [i] Starting FreqTrade Dry-Run in this window.
ECHO [i] Logs are also written to: user_data\logs\freqtrade_dryrun.log
ECHO [i] FreqUI Dashboard: http://127.0.0.1:8080/
ECHO [i] Login: freqtrader / ProductionHighlySecurePassword2026!
ECHO.
ECHO [i] Press Ctrl+C once to stop the bot gracefully.
ECHO.

:: Run FreqTrade in THIS window so errors and the training loop are visible.
:: Logs are mirrored to a dedicated dry-run log file.
python -m freqtrade trade !CONFIG_ARGS! !TRADE_ARGS! --dry-run --dry-run-wallet 10000 --logfile user_data\logs\freqtrade_dryrun.log

ECHO.
ECHO [i] FreqTrade stopped. Review the log above or user_data\logs\freqtrade_dryrun.log.
ECHO.
PAUSE
