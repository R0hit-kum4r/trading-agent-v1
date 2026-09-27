import os
import gzip
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

IST = ZoneInfo("Asia/Kolkata")

STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "SBIN", "ITC", "BHARTIARTL", "LT", "AXISBANK",
    "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI",
    "SUNPHARMA", "TITAN", "ADANIENT", "ADANIPORTS",
    "ASIANPAINT", "HCLTECH", "WIPRO", "TECHM",
    "ULTRACEMCO", "M&M", "NTPC", "POWERGRID",
    "TATASTEEL", "COALINDIA", "JSWSTEEL", "ONGC"
]

SWING_LEFT = 5
SWING_RIGHT = 5
MAX_BOS_WAIT = 12
MIN_RR = 3.0
SL_BUFFER = 0.001  # 0.10%

UPSTOX_TOKEN = os.environ["UPSTOX_TOKEN"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
UPSTOX_BASE = "https://api.upstox.com/v3"
STATE_FILE = "state/telegram_alert_state.json"


def normalize_timestamps(series):
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(IST).dt.tz_localize(None)
    return s


def get_instrument_map():
    r = requests.get(INSTRUMENT_URL, timeout=30)
    r.raise_for_status()
    instruments = json.loads(gzip.decompress(r.content).decode("utf-8"))

    mapping = {}
    for item in instruments:
        if (
            item.get("segment") == "NSE_EQ"
            and item.get("instrument_type") == "EQ"
            and item.get("trading_symbol")
            and item.get("instrument_key")
        ):
            mapping[item["trading_symbol"]] = item["instrument_key"]

    missing = [s for s in STOCKS if s not in mapping]
    if missing:
        raise RuntimeError(f"Missing NSE instruments: {missing}")

    return {s: mapping[s] for s in STOCKS}


def candles_to_df(candles):
    if not candles:
        return pd.DataFrame()

    rows = []
    for c in candles:
        rows.append({
            "timestamp": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "open_interest": float(c[6]) if len(c) > 6 else 0.0,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = normalize_timestamps(df["timestamp"])
    return (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )


def fetch_historical_15m(instrument_key, to_date):
    end_date = datetime.strptime(to_date, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=30)

    url = (
        f"{UPSTOX_BASE}/historical-candle/"
        f"{instrument_key}/minutes/15/"
        f"{to_date}/{start_date.isoformat()}"
    )

    r = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_TOKEN}",
        },
        timeout=30,
    )
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return candles_to_df(candles)


def fetch_current_day_15m(instrument_key):
    url = (
        f"{UPSTOX_BASE}/historical-candle/intraday/"
        f"{instrument_key}/minutes/15"
    )

    r = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {UPSTOX_TOKEN}",
        },
        timeout=30,
    )
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return candles_to_df(candles)


def build_context(df):
    df = df.copy()
    df["timestamp"] = normalize_timestamps(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    one_hour = []
    four_hour = []

    for _, day in df.groupby("date", sort=True):
        day = day.sort_values("timestamp").reset_index(drop=True)

        # 1H = 4 consecutive 15M candles.
        for start in range(0, len(day) - 3, 4):
            block = day.iloc[start:start + 4]

            if len(block) != 4:
                continue

            if not all(
                d == timedelta(minutes=15)
                for d in block["timestamp"].diff().dropna()
            ):
                continue

            one_hour.append({
                "timestamp": block["timestamp"].iloc[-1],
                "open": block["open"].iloc[0],
                "high": block["high"].max(),
                "low": block["low"].min(),
                "close": block["close"].iloc[-1],
                "volume": block["volume"].sum(),
                "open_interest": block["open_interest"].iloc[-1],
            })

        # Session-aware 4H baseline:
        # first complete 16 x 15M block, 09:15 -> 13:00 timestamp.
        block = day.iloc[:16]

        if len(block) == 16 and all(
            d == timedelta(minutes=15)
            for d in block["timestamp"].diff().dropna()
        ):
            four_hour.append({
                "timestamp": block["timestamp"].iloc[-1],
                "open": block["open"].iloc[0],
                "high": block["high"].max(),
                "low": block["low"].min(),
                "close": block["close"].iloc[-1],
                "volume": block["volume"].sum(),
                "open_interest": block["open_interest"].iloc[-1],
            })

    return pd.DataFrame(one_hour), pd.DataFrame(four_hour)


def build_structure(df):
    df = df.copy().reset_index(drop=True)

    if df.empty:
        df["bias"] = pd.Series(dtype="object")
        return df

    df["timestamp"] = normalize_timestamps(df["timestamp"])
    df["swing_high"] = False
    df["swing_low"] = False
    df["swing_high_confirmed_at"] = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )
    df["swing_low_confirmed_at"] = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )

    if len(df) >= SWING_LEFT + SWING_RIGHT + 1:
        for i in range(SWING_LEFT, len(df) - SWING_RIGHT):
            h = df.loc[i, "high"]
            l = df.loc[i, "low"]

            if (
                h > df.loc[i-SWING_LEFT:i-1, "high"].max()
                and h > df.loc[i+1:i+SWING_RIGHT, "high"].max()
            ):
                df.loc[i, "swing_high"] = True
                df.loc[i, "swing_high_confirmed_at"] = df.loc[
                    i + SWING_RIGHT, "timestamp"
                ]

            if (
                l < df.loc[i-SWING_LEFT:i-1, "low"].min()
                and l < df.loc[i+1:i+SWING_RIGHT, "low"].min()
            ):
                df.loc[i, "swing_low"] = True
                df.loc[i, "swing_low_confirmed_at"] = df.loc[
                    i + SWING_RIGHT, "timestamp"
                ]

    df["bias"] = "Neutral"

    for i in range(len(df)):
        now = df.loc[i, "timestamp"]

        highs = df[
            (df["swing_high"])
            & df["swing_high_confirmed_at"].notna()
            & (df["swing_high_confirmed_at"] <= now)
        ]

        lows = df[
            (df["swing_low"])
            & df["swing_low_confirmed_at"].notna()
            & (df["swing_low_confirmed_at"] <= now)
        ]

        if len(highs) < 2 or len(lows) < 2:
            continue

        latest_high = highs.iloc[-1]["high"]
        previous_high = highs.iloc[-2]["high"]
        latest_low = lows.iloc[-1]["low"]
        previous_low = lows.iloc[-2]["low"]

        if latest_high > previous_high and latest_low > previous_low:
            df.loc[i, "bias"] = "Bullish"
        elif latest_high < previous_high and latest_low < previous_low:
            df.loc[i, "bias"] = "Bearish"

    return df


def get_bias_at(df, timestamp):
    if df is None or df.empty:
        return "Neutral"

    temp = df[df["timestamp"] < timestamp]

    if temp.empty:
        return "Neutral"

    return str(temp.iloc[-1]["bias"])


def find_signals(df15, h1_structure, h4_structure, latest_candle_time):
    df = df15.copy().reset_index(drop=True)
    df["timestamp"] = normalize_timestamps(df["timestamp"])

    if len(df) < SWING_LEFT + SWING_RIGHT + 5:
        return []

    df["swing_high"] = False
    df["swing_low"] = False
    df["swing_high_confirmed_at"] = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )
    df["swing_low_confirmed_at"] = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )

    for i in range(SWING_LEFT, len(df) - SWING_RIGHT):
        h = df.loc[i, "high"]
        l = df.loc[i, "low"]

        if (
            h > df.loc[i-SWING_LEFT:i-1, "high"].max()
            and h > df.loc[i+1:i+SWING_RIGHT, "high"].max()
        ):
            df.loc[i, "swing_high"] = True
            df.loc[i, "swing_high_confirmed_at"] = df.loc[
                i + SWING_RIGHT, "timestamp"
            ]

        if (
            l < df.loc[i-SWING_LEFT:i-1, "low"].min()
            and l < df.loc[i+1:i+SWING_RIGHT, "low"].min()
        ):
            df.loc[i, "swing_low"] = True
            df.loc[i, "swing_low_confirmed_at"] = df.loc[
                i + SWING_RIGHT, "timestamp"
            ]

    latest_rows = df[df["timestamp"] == latest_candle_time]

    if latest_rows.empty:
        return []

    latest_idx = latest_rows.index[-1]
    signals = []

    for i in range(len(df)):
        now = df.loc[i, "timestamp"]

        highs = df[
            (df["swing_high"])
            & df["swing_high_confirmed_at"].notna()
            & (df["swing_high_confirmed_at"] < now)
        ]

        lows = df[
            (df["swing_low"])
            & df["swing_low_confirmed_at"].notna()
            & (df["swing_low_confirmed_at"] < now)
        ]

        # BUY: sweep below latest confirmed low, then frozen BOS
        # above the latest confirmed high.
        if not lows.empty and not highs.empty:
            sweep_level = float(lows.iloc[-1]["low"])

            if (
                df.loc[i, "low"] < sweep_level
                and df.loc[i, "close"] > sweep_level
            ):
                sweep_extreme = float(df.loc[i, "low"])
                bos_level = float(highs.iloc[-1]["high"])
                end_idx = min(i + MAX_BOS_WAIT, len(df) - 1)

                for j in range(i + 1, end_idx + 1):
                    if float(df.loc[j, "close"]) > bos_level:
                        # We only alert when this BOS is the newest
                        # completed 15M candle.
                        if j != latest_idx:
                            break

                        entry = float(df.loc[j, "close"])
                        sl = sweep_extreme * (1 - SL_BUFFER)
                        risk = entry - sl

                        if risk <= 0:
                            break

                        target = entry + risk * MIN_RR
                        bos_time = df.loc[j, "timestamp"]
                        bias_1h = get_bias_at(h1_structure, bos_time)
                        bias_4h = get_bias_at(h4_structure, bos_time)

                        if bias_1h == "Bullish" and bias_4h == "Bullish":
                            signals.append({
                                "direction": "BUY",
                                "sweep_time": df.loc[i, "timestamp"],
                                "bos_time": bos_time,
                                "entry": entry,
                                "stop_loss": sl,
                                "target": target,
                                "risk": risk,
                                "1H_bias": bias_1h,
                                "4H_bias": bias_4h,
                                "sweep_level": sweep_level,
                                "sweep_extreme": sweep_extreme,
                                "bos_level": bos_level,
                                "bars_sweep_to_bos": j - i,
                            })
                        break

        # SELL: sweep above latest confirmed high, then frozen BOS
        # below the latest confirmed low.
        if not highs.empty and not lows.empty:
            sweep_level = float(highs.iloc[-1]["high"])

            if (
                df.loc[i, "high"] > sweep_level
                and df.loc[i, "close"] < sweep_level
            ):
                sweep_extreme = float(df.loc[i, "high"])
                bos_level = float(lows.iloc[-1]["low"])
                end_idx = min(i + MAX_BOS_WAIT, len(df) - 1)

                for j in range(i + 1, end_idx + 1):
                    if float(df.loc[j, "close"]) < bos_level:
                        if j != latest_idx:
                            break

                        entry = float(df.loc[j, "close"])
                        sl = sweep_extreme * (1 + SL_BUFFER)
                        risk = sl - entry

                        if risk <= 0:
                            break

                        target = entry - risk * MIN_RR
                        bos_time = df.loc[j, "timestamp"]
                        bias_1h = get_bias_at(h1_structure, bos_time)
                        bias_4h = get_bias_at(h4_structure, bos_time)

                        if bias_1h == "Bearish" and bias_4h == "Bearish":
                            signals.append({
                                "direction": "SELL",
                                "sweep_time": df.loc[i, "timestamp"],
                                "bos_time": bos_time,
                                "entry": entry,
                                "stop_loss": sl,
                                "target": target,
                                "risk": risk,
                                "1H_bias": bias_1h,
                                "4H_bias": bias_4h,
                                "sweep_level": sweep_level,
                                "sweep_extreme": sweep_extreme,
                                "bos_level": bos_level,
                                "bars_sweep_to_bos": j - i,
                            })
                        break

    return signals


def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return set()


def save_state(seen):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, indent=2)


def signal_id(signal):
    return f"{signal['stock']}|{signal['direction']}|{signal['bos_time']}"


def send_telegram(signal):
    message = (
        ("🟢" if signal["direction"] == "BUY" else "🔴")
        + " PAPER TRADE ALERT\n\n"
        f"Stock: {signal['stock']}\n"
        f"Direction: {signal['direction']}\n\n"
        f"Entry: ₹{signal['entry']:.2f}\n"
        f"Stop Loss: ₹{signal['stop_loss']:.2f}\n"
        f"Target: ₹{signal['target']:.2f}\n"
        f"Risk: ₹{signal['risk']:.2f}\n"
        f"Reward: 3R\n\n"
        f"4H Bias: {signal['4H_bias']}\n"
        f"1H Bias: {signal['1H_bias']}\n"
        "Liquidity Sweep: YES\n"
        "15M BOS: YES\n\n"
        "⚠️ PAPER ALERT ONLY\n"
        "🤖 Auto Trading: OFF"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=15,
    )
    r.raise_for_status()


def latest_completed_15m(now):
    minute_block = (now.minute // 15) * 15
    boundary = now.replace(
        minute=minute_block, second=0, microsecond=0
    )
    return boundary - timedelta(minutes=15)


def market_is_open(now):
    if now.weekday() >= 5:
        return False

    start = now.replace(
        hour=9, minute=15, second=0, microsecond=0
    )
    end = now.replace(
        hour=15, minute=30, second=0, microsecond=0
    )

    return start <= now <= end


def scan():
    now = datetime.now(IST)

    if not market_is_open(now):
        print(f"Market closed: {now:%Y-%m-%d %H:%M:%S %Z}")
        return

    latest = latest_completed_15m(now)
    market_open = now.replace(
        hour=9, minute=15, second=0, microsecond=0
    )

    if latest < market_open:
        print("No completed market candle yet.")
        return

    print("=" * 70)
    print("STRATEGY V1 PAPER SCANNER")
    print(f"Run: {now:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Latest completed 15M candle: {latest}")
    print("Auto Trading: OFF")
    print("=" * 70)

    instruments = get_instrument_map()
    seen = load_state()
    changed = False
    total_signals = 0
    sent = 0

    for n, stock in enumerate(STOCKS, 1):
        try:
            today = now.strftime("%Y-%m-%d")

            historical = fetch_historical_15m(
                instruments[stock], today
            )

            current = fetch_current_day_15m(
                instruments[stock]
            )

            if historical.empty and current.empty:
                print(f"{n:02d}/30 {stock:<15} -> no data")
                continue

            # Historical endpoint can overlap today's candles.
            historical = historical[
                historical["timestamp"].dt.strftime("%Y-%m-%d") != today
            ]

            full = pd.concat(
                [historical, current],
                ignore_index=True
            )

            full = (
                full.drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # Never use an incomplete 15M candle.
            full = full[full["timestamp"] <= latest].copy()

            if full.empty or not (full["timestamp"] == latest).any():
                print(
                    f"{n:02d}/30 {stock:<15} -> latest candle unavailable"
                )
                continue

            h1, h4 = build_context(full)
            h1_structure = build_structure(h1)
            h4_structure = build_structure(h4)

            signals = find_signals(
                full,
                h1_structure,
                h4_structure,
                latest,
            )

            total_signals += len(signals)

            for signal in signals:
                signal["stock"] = stock
                sid = signal_id(signal)

                if sid in seen:
                    continue

                print(
                    f"🚨 NEW {stock} "
                    f"{signal['direction']} "
                    f"{signal['bos_time']}"
                )

                send_telegram(signal)
                seen.add(sid)
                changed = True
                sent += 1

            print(
                f"{n:02d}/30 {stock:<15} "
                f"-> {len(signals)} signal(s)"
            )

        except Exception as exc:
            print(
                f"{n:02d}/30 {stock:<15} "
                f"-> ERROR: {exc}"
            )

    if changed:
        save_state(seen)

    print("-" * 70)
    print(f"Signals on latest candle: {total_signals}")
    print(f"New Telegram alerts: {sent}")
    print("SCAN COMPLETE")


if __name__ == "__main__":
    scan()
