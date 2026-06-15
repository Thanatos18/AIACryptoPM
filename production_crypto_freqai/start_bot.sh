#!/usr/bin/env bash
# Usage:
#   ./start_bot.sh          -> Dry-run (paper trading)
#   ./start_bot.sh --live   -> Live trading (real orders)
#
# For one-click dry-run with auto data download, use start_dryrun.sh instead.

set -e

cd "$(dirname "$0")"

MODE="DRY_RUN"
CONFIG_ARGS="-c config.json -c freqai_config.json"
TRADE_ARGS="--strategy FreqAiAdaptiveRollingStrategy --user-data-dir user_data --strategy-path user_data/strategies --freqaimodel LightGBMRegressorCPU --freqaimodel-path user_data/freqai_models"

for arg in "$@"; do
    if [ "$arg" == "--live" ]; then
        MODE="LIVE"
        CONFIG_ARGS="-c config.json -c freqai_config.json -c config.live.json"
    fi
done

echo "====================================================================="
echo " Production Crypto Trading System (FreqTrade + FreqAI)"
echo " Mode: $MODE"
echo "====================================================================="

if [ -d ".venv" ]; then
    echo "[i] Activating Python virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[i] Activating Python virtual environment..."
    source venv/bin/activate
else
    echo "[!] No virtual environment found (.venv). Ensure dependencies are installed."
fi

mkdir -p user_data/logs user_data/models user_data/data user_data/freqai_models

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

if [ "$MODE" == "LIVE" ]; then
    echo "[!] WARNING: LIVE trading mode — real capital at risk!"
    echo "[!] Ensure exchange API keys in config.json are correct."
    freqtrade trade \
        $CONFIG_ARGS \
        $TRADE_ARGS
else
    echo "[i] DRY-RUN mode — simulated trades only."
    echo "[i] Tip: use ./start_dryrun.sh for one-click data download + dry-run."
    freqtrade trade \
        $CONFIG_ARGS \
        $TRADE_ARGS \
        --dry-run \
        --dry-run-wallet 10000
fi
