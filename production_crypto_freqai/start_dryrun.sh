#!/usr/bin/env bash
# One-click dry-run launcher (WSL2 / Linux):
#   1. Turn on WARP (or ensure Binance is reachable)
#   2. ./start_dryrun.sh
#   3. Bot downloads data, trains FreqAI models, and starts paper trading

set -e

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

cd "$(dirname "$0")"

echo "====================================================================="
echo " FreqTrade + FreqAI  |  ONE-CLICK DRY-RUN"
echo " Simulated wallet: \$10,000 USDT  |  Timeframe: 15m"
echo "====================================================================="
echo ""
echo " Step 1: Enable WARP (or your VPN) so Binance data can be fetched."
echo " Step 2: This script will download data, then start paper trading."
echo ""

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[!] Virtual environment not found. Run setup first."
    exit 1
fi

mkdir -p user_data/logs user_data/models user_data/data user_data/freqai_models

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

CONFIG_ARGS="-c config.json -c freqai_config.json --user-data-dir user_data"
TRADE_ARGS="--strategy FreqAiAdaptiveRollingStrategy --strategy-path user_data/strategies --freqaimodel LightGBMClassifierCPU --freqaimodel-path user_data/freqai_models"

echo ""
echo "[i] Downloading / updating market data (60 days, 15m + 1h)..."
echo "[i] If this step hangs, confirm WARP is connected."
echo ""

if ! freqtrade download-data \
    $CONFIG_ARGS \
    --exchange binance \
    --pairs BTC/USDT ETH/USDT SOL/USDT LINK/USDT AVAX/USDT BNB/USDT \
    --timeframes 15m 1h \
    --days 60 \
    --prepend; then
    echo ""
    echo "[!] Data download failed. Check WARP/network, then re-run this script."
    echo "[!] Attempting to start bot with any existing local data..."
    echo ""
    sleep 5
fi

echo ""
echo "[i] Starting FreqTrade dry-run..."
echo "[i] Dashboard: http://127.0.0.1:8080/"
echo "[i] Press Ctrl+C to stop the bot."
echo ""

freqtrade trade \
    $CONFIG_ARGS \
    $TRADE_ARGS \
    --dry-run \
    --dry-run-wallet 10000
