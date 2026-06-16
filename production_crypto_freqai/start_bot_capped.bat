@echo off
SETLOCAL EnableDelayedExpansion
cd /d "%~dp0"

:: Requires the `procgov` tool (Process Governor by Lowleveldesign).
:: Install once with: winget install procgov
:: This caps the python.exe child to 10 GB RSS and 6 cores.

procgov.exe --maxmem 10G --maxcpurate 75 --recursive ^
    cmd /c "CALL .venv\Scripts\activate.bat && freqtrade trade -c config.json -c freqai_config.json --strategy FreqAiAdaptiveRollingStrategy --user-data-dir user_data --strategy-path user_data\strategies --freqaimodel LightGBMClassifierCPU --freqaimodel-path user_data\freqai_models --dry-run --dry-run-wallet 10000"
