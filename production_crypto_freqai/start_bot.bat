@echo off
SETLOCAL EnableDelayedExpansion
chcp 65001 >nul 2>&1

SET OMP_NUM_THREADS=4
SET OPENBLAS_NUM_THREADS=4
SET MKL_NUM_THREADS=4
SET NUMEXPR_NUM_THREADS=4

TITLE Production FreqTrade + FreqAI Trading System (16GB RAM CPU Optimized)

:: Usage:
::   start_bot.bat          -> Dry-run (paper trading)
::   start_bot.bat --live   -> Live trading (real orders)
::
:: For one-click dry-run with auto data download, use START_DRYRUN.bat instead.

cd /d "%~dp0"

SET MODE=DRY_RUN
SET CONFIG_ARGS=-c config.json -c freqai_config.json
SET TRADE_ARGS=--strategy FreqAiAdaptiveRollingStrategy --user-data-dir user_data --strategy-path user_data\strategies --freqaimodel LightGBMClassifierCPU --freqaimodel-path user_data\freqai_models

IF "%~1"=="--live" (
    SET MODE=LIVE
    SET CONFIG_ARGS=-c config.json -c freqai_config.json -c config.live.json
)

ECHO =====================================================================
ECHO  Production Crypto Trading System (FreqTrade + FreqAI)
ECHO  Mode: %MODE%
ECHO =====================================================================

IF EXIST ".venv\Scripts\activate.bat" (
    ECHO [i] Activating Python virtual environment (.venv)...
    CALL .venv\Scripts\activate.bat
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    ECHO [i] Activating Python virtual environment (venv)...
    CALL venv\Scripts\activate.bat
) ELSE (
    ECHO [!] No virtual environment found. Running in system Python.
)

IF NOT EXIST "user_data\logs" MKDIR "user_data\logs"
IF NOT EXIST "user_data\models" MKDIR "user_data\models"
IF NOT EXIST "user_data\data" MKDIR "user_data\data"
IF NOT EXIST "user_data\freqai_models" MKDIR "user_data\freqai_models"

SET PYTHONIOENCODING=utf-8
SET PYTHONUTF8=1

IF "%MODE%"=="LIVE" (
    ECHO [!] WARNING: LIVE trading mode — real capital at risk!
    ECHO [!] Ensure exchange API keys in config.json are correct.
    freqtrade trade %CONFIG_ARGS% %TRADE_ARGS%
) ELSE (
    ECHO [i] DRY-RUN mode — simulated trades only.
    ECHO [i] Tip: use START_DRYRUN.bat for one-click data download + dry-run.
    freqtrade trade %CONFIG_ARGS% %TRADE_ARGS% --dry-run --dry-run-wallet 10000
)

PAUSE
