# Trading Agent V1 — NSE Paper Alert Bot

Paper-alert scanner for 30 NSE stocks.

## Strategy

- 4H major bias
- 1H market structure
- 15M liquidity sweep
- 15M frozen BOS
- Minimum 3R target
- 0.10% SL buffer beyond sweep extreme
- Telegram alert only
- Auto trading OFF

## GitHub Secrets

Create these repository secrets:

- `UPSTOX_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Do not put tokens in `scanner.py`.

## Files

- `scanner.py` — scanner and Telegram alert logic
- `requirements.txt` — Python dependencies
- `.github/workflows/paper_scanner.yml` — scheduled GitHub Actions workflow
- `state/telegram_alert_state.json` — duplicate-alert state

## Schedule

The workflow is scheduled around every 15 minutes during NSE hours using UTC cron. GitHub Actions can delay scheduled jobs, so the Python script independently selects the latest completed 15-minute candle.

You can also start it manually from:

GitHub → Actions → NSE Strategy V1 Paper Scanner → Run workflow

## Safety

This project sends paper alerts only. It does not place broker orders.
