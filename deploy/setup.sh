#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# One-command setup for Oracle Cloud / Ubuntu VM
# Usage: ssh into your VM, clone the repo, then run:
#   bash deploy/setup.sh
# ============================================================

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="trading-bot"

echo "==> Installing Python dependencies..."
sudo apt update -qq
sudo apt install -y python3-pip python3-venv

echo "==> Creating virtual environment..."
python3 -m venv "$APP_DIR/.venv"
source "$APP_DIR/.venv/bin/activate"
pip install --upgrade pip
pip install ccxt pandas numpy ta pyyaml python-dotenv schedule colorama

echo "==> Installing systemd service..."
sudo cp "$APP_DIR/deploy/trading-bot.service" /etc/systemd/system/
# Patch paths and user to match this machine
sudo sed -i "s|/home/ubuntu/crypto-trading-bot|$APP_DIR|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|User=ubuntu|User=$(whoami)|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|/usr/bin/python3|$APP_DIR/.venv/bin/python3|g" /etc/systemd/system/trading-bot.service

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env && nano .env   # add your API keys"
echo "  2. nano config.yaml                     # adjust settings"
echo "  3. sudo systemctl start trading-bot     # start the bot"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status trading-bot       # check status"
echo "  sudo journalctl -u trading-bot -f       # live logs"
echo "  sudo systemctl restart trading-bot      # restart"
echo "  sudo systemctl stop trading-bot         # stop"
