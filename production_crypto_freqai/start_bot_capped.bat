@echo off
SETLOCAL EnableDelayedExpansion
cd /d "%~dp0"

:: Requires the `procgov` tool (Process Governor by Lowleveldesign).
:: Install once with: winget install procgov
:: This caps the python.exe child to 10 GB RSS and 6 cores.

SET CONFIG_ARGS=-c config.json -c freqai_config.json
IF EXIST config.local.json (
    SET CONFIG_ARGS=!CONFIG_ARGS! -c config.local.json
)

procgov.exe --maxmem 10G --maxcpurate 75 --recursive ^
    cmd /c "CALL .venv\Scripts\activate.bat && python -m freqtrade trade !CONFIG_ARGS! --strategy FreqAiAdaptiveRollingStrategy --user-data-dir user_data --strategy-path user_data\strategies --freqaimodel LightGBMClassifierCPU --freqaimodel-path user_data\freqai_models --dry-run --dry-run-wallet 10000"
