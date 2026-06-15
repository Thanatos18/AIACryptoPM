# Windows & WSL2 Production Setup Guide for FreqTrade + FreqAI

This document provides definitive, step-by-step instructions for deploying the automated FreqTrade + FreqAI trading system on a Windows laptop with 16GB RAM and CPU-only processing.

---

## 1. Architectural Overview: Native Windows vs. WSL2

While FreqTrade can run natively on Windows via Python, **we strongly recommend executing the system inside WSL2 (Windows Subsystem for Linux)**. 

### Why WSL2 is Superior for Production Machine Learning:
1. **File System Performance**: Python's multi-threaded model and SQLite disk operations run significantly faster on a native Linux EXT4 file system than on Windows NTFS.
2. **C++ Compilation**: Libraries like `LightGBM`, `CatBoost`, and `TA-Lib` compile and execute with dramatically better memory management and SIMD vectorization under Ubuntu/Debian.
3. **Process Isolation**: WSL2 allows you to enforce precise memory limits via a `.wslconfig` file, completely eliminating the risk of the trading bot starving your Windows OS of RAM and causing system freezes.

---

## 2. WSL2 Installation & RAM Limiting (Highly Recommended)

### Step 2.1: Install WSL2
Open your Windows PowerShell **as Administrator** and execute:
```powershell
wsl --install -d Ubuntu-24.04
```
*If WSL is already installed, update it to ensure you have the latest WSL2 kernel:*
```powershell
wsl --update
```
Restart your computer if prompted.

### Step 2.2: Enforce the 16GB RAM Safeguard
To guarantee that WSL2 shares memory elegantly with your Windows host OS, configure a `.wslconfig` file.
In Windows, press `Win + R`, type `%userprofile%` and press Enter. Create a new file named exactly `.wslconfig` and add the following content:

```ini
[wsl2]
memory=12GB
processors=6
swap=4GB
localhostForwarding=true
```
*This configures WSL2 to use a maximum of 12GB RAM (leaving 4GB entirely for Windows), allocates 6 CPU cores, sets up a 4GB SSD swap file to prevent unexpected OOM crashes, and automatically forwards localhost ports so you can view FreqUI from your native Windows browser.*

Restart WSL to apply the limits:
```powershell
wsl --shutdown
```

---

## 3. System Dependencies & Python Setup inside Ubuntu

Open your **Ubuntu 24.04** terminal from the Windows Start menu and run:

### Step 3.1: Install Core Build Tools & SQLite
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-dev build-essential git libssl-dev cmake curl sqlite3 libsqlite3-dev
```

### Step 3.2: Clone / Move the Production Repository
If you haven't already placed your project files in your Linux home directory, copy them from your Windows drive (which is mounted at `/mnt/c/`):
```bash
cp -r /mnt/c/Users/YOUR_WINDOWS_USERNAME/production_crypto_freqai ~/production_crypto_freqai
cd ~/production_crypto_freqai
```

### Step 3.3: Create a Virtual Environment & Install Packages
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

---

## 4. Telegram Bot Configuration

FreqTrade includes incredible native Telegram integration that provides real-time alerts, trade summaries, and interactive control over your bot (e.g., forcing exits, checking balances, stopping the bot).

### Step 4.1: Create Your Bot via BotFather
1. Open Telegram on your phone or PC and search for **`@BotFather`**.
2. Send the command `/newbot`.
3. Provide a friendly name (e.g., `FreqAI Production Trader`) and a unique bot username (e.g., `MyFreqAiProductionBot`).
4. `@BotFather` will generate an **API Token** (a long string like `1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ`). **Copy this token.**

### Step 4.2: Get Your Personal Chat ID
1. Search for **`@userinfobot`** or **`@IDBot`** in Telegram and send the message `/start`.
2. It will reply with your personal **Chat ID** (a series of numbers like `987654321`). **Copy this ID.**

### Step 4.3: Enter Credentials into `config.json`
Open `config.json` in your project directory and update the `telegram` section:
```json
"telegram": {
    "enabled": true,
    "token": "1234567890:ABCdefGhIJKlmNoPQRsTUVwxyZ",
    "chat_id": "987654321"
}
```

---

## 5. FreqUI (Web UI) Dashboard Access

FreqTrade features a beautiful local web visualizer called **FreqUI**.

### Step 5.1: Review Settings in `config.json`
Our supplied `config.json` is fully configured for FreqUI:
```json
"api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "ProductionSuperSecretKey_CHANGE_ME",
    "ws_token": "ProductionWsToken_CHANGE_ME",
    "CORS_origins": []
}
```

### Step 5.2: Accessing FreqUI in Your Browser
Once you start the bot (see Section 6), open your preferred web browser in Windows (Chrome, Firefox, Brave) and navigate to:
```
http://127.0.0.1:8080/
```
Or if accessing across your local Wi-Fi network:
```
http://<YOUR_LAPTOP_LOCAL_IP>:8080/
```

**First Login Credentials**:
* **Username**: `freqtrader`
* **Password**: `ProductionHighlySecurePassword2026!` (As defined in your `config.json` under `initial_state`)

Here you can view open trades, live equity curves, performance per pair, and system logs.

---

## 6. Execution: Dry-Run vs. Live Trading Modes

Our production scripts allow you to safely test the bot in simulated paper-trading mode (**Dry-Run**) before committing real capital.

### Step 6.1: Using the Startup Scripts

#### On WSL2 / Linux (`start_bot.sh`):
Make the script executable first:
```bash
chmod +x start_bot.sh
```
* **Launch in Dry-Run Mode** (Safely paper trades against live orderbooks):
  ```bash
  ./start_bot.sh
  ```
* **Launch in Live Trading Mode** (Executes actual trades with real capital):
  ```bash
  ./start_bot.sh --live
  ```

#### On Native Windows (`start_bot.bat`):
* **Launch in Dry-Run Mode**: Simply double-click `start_bot.bat` or run `start_bot.bat` in Command Prompt.
* **Launch in Live Trading Mode**: Open Command Prompt and execute:
  ```cmd
  start_bot.bat --live
  ```

### Step 6.2: Entering Exchange API Credentials for Live Mode
Before switching to live trading, you must provide your real API keys from Binance or your chosen exchange.
In `config.json`, locate the `exchange` block:
```json
"exchange": {
    "name": "binance",
    "key": "YOUR_REAL_BINANCE_API_KEY",
    "secret": "YOUR_REAL_BINANCE_SECRET_KEY",
    "ccxt_config": {
        "enableRateLimit": true
    },
    "ccxt_async_config": {
        "enableRateLimit": true
    }
}
```
*Note: Make sure your exchange API Key has **Spot Trading enabled** and **Withdrawals disabled** for absolute security.*

---

## 7. Running the Supplementary CPU Analytics Dashboard

In addition to FreqUI, we provide a custom, lightweight Python analytics app that performs deep diagnostic analysis on your trading database (`tradesv3.sqlite`) and FreqAI performance.

To start the supplementary analytics dashboard, run:
```bash
source .venv/bin/activate
streamlit run scripts/cpu_analytics_dashboard.py
```
This will automatically launch a beautifully formatted interactive web dashboard in your browser showing:
1. **Interactive Equity Curves & Max Drawdown**
2. **Win Rate Trends & Sharpe Ratios**
3. **Asset-by-Asset PnL Breakdown**
4. **FreqAI ML Feature Importance Tracking**
