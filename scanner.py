import os
import json
import gzip
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


# =========================================================
# CONFIG
# =========================================================

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
SL_BUFFER = 0.001

UPSTOX_TOKEN = os.environ["UPSTOX_TOKEN"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

UPSTOX_BASE = "https://api.upstox.com"
INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/"
    "instruments/exchange/NSE.json.gz"
)


# =========================================================
# INSTRUMENT MAP
# =========================================================

def get_instrument_map():
    response = requests.get(INSTRUMENT_URL, timeout=30)
    response.raise_for_status()

    raw = gzip.decompress(response.content)
    data = json.loads(raw.decode("utf-8"))

    mapping = {}

    for item in data:
        if (
            item.get("segment") == "NSE_EQ"
            and item.get("instrument_type") == "EQ"
        ):
            symbol = item.get("trading_symbol")
            key = item.get("instrument_key")

            if symbol and key:
                mapping[symbol] = key

    missing = [s for s in STOCKS if s not in mapping]

    if missing:
        raise RuntimeError(f"Missing instruments: {missing}")

    return {s: mapping[s] for s in STOCKS}


# =========================================================
# HISTORICAL 15M DATA
# =========================================================

def fetch_15m(stock, instrument_key, today):
    from_date = today - timedelta(days=30)

    url = (
        f"{UPSTOX_BASE}/v3/historical-candle/"
        f"{instrument_key}/minutes/15/"
        f"{today.isoformat()}/"
        f"{from_date.isoformat()}"
    )

    headers = {
        "Authorization": f"Bearer {UPSTOX_TOKEN}",
        "Accept": "application/json"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    payload = response.json()
    candles = payload.get("data", {}).get("candles", [])

    if not candles:
        return pd.DataFrame()

    rows = []

    for c in candles:
        rows.append({
            "timestamp": pd.to_datetime(c[0]).tz_convert(IST).tz_localize(None),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
            "open_interest": float(c[6]) if len(c) > 6 else 0
        })

    df = pd.DataFrame(rows)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)

    return df


# =========================================================
# SESSION DATA
# =========================================================

def build_1h(df):
    df = df.copy()

    df["date"] = df["timestamp"].dt.date

    result = []

    for _, day in df.groupby("date", sort=True):

        day = day.sort_values("timestamp").reset_index(drop=True)

        # 4 x 15M = 1H
        for i in range(0, len(day) - 3, 4):

            block = day.iloc[i:i+4]

            if len(block) != 4:
                continue

            # Ensure consecutive 15M candles
            diffs = block["timestamp"].diff().dropna()

            if not all(d == timedelta(minutes=15) for d in diffs):
                continue

            result.append({
                "timestamp": block.iloc[-1]["timestamp"],
                "close_time": block.iloc[-1]["timestamp"]
                         + timedelta(minutes=15),
                "open": block.iloc[0]["open"],
                "high": block["high"].max(),
                "low": block["low"].min(),
                "close": block.iloc[-1]["close"],
                "volume": block["volume"].sum()
            })

    return pd.DataFrame(result)


def build_4h(df):
    df = df.copy()

    df["date"] = df["timestamp"].dt.date

    result = []

    for _, day in df.groupby("date", sort=True):

        day = day.sort_values("timestamp").reset_index(drop=True)

        # Only complete 09:15 -> 13:15 block
        if len(day) < 16:
            continue

        block = day.iloc[:16]

        diffs = block["timestamp"].diff().dropna()

        if not all(d == timedelta(minutes=15) for d in diffs):
            continue

        result.append({
            "timestamp": block.iloc[-1]["timestamp"],
            "close_time": block.iloc[-1]["timestamp"]
                         + timedelta(minutes=15),
            "open": block.iloc[0]["open"],
            "high": block["high"].max(),
            "low": block["low"].min(),
            "close": block.iloc[-1]["close"],
            "volume": block["volume"].sum()
        })

    return pd.DataFrame(result)


# =========================================================
# SWING DETECTION
# =========================================================

def add_swings(df):
    df = df.copy()

    df["swing_high"] = False
    df["swing_low"] = False
    df["swing_high_level"] = np.nan
    df["swing_low_level"] = np.nan
    df["swing_high_confirmed_at"] = pd.NaT
    df["swing_low_confirmed_at"] = pd.NaT

    for i in range(
        SWING_LEFT,
        len(df) - SWING_RIGHT
    ):

        high = df.iloc[i]["high"]
        low = df.iloc[i]["low"]

        left_highs = df.iloc[
            i-SWING_LEFT:i
        ]["high"]

        right_highs = df.iloc[
            i+1:i+SWING_RIGHT+1
        ]["high"]

        left_lows = df.iloc[
            i-SWING_LEFT:i
        ]["low"]

        right_lows = df.iloc[
            i+1:i+SWING_RIGHT+1
        ]["low"]

        if high > left_highs.max() and high > right_highs.max():

            confirm_idx = i + SWING_RIGHT

            df.loc[i, "swing_high"] = True
            df.loc[i, "swing_high_level"] = high
            df.loc[i, "swing_high_confirmed_at"] = (
                df.iloc[confirm_idx]["timestamp"]
            )

        if low < left_lows.min() and low < right_lows.min():

            confirm_idx = i + SWING_RIGHT

            df.loc[i, "swing_low"] = True
            df.loc[i, "swing_low_level"] = low
            df.loc[i, "swing_low_confirmed_at"] = (
                df.iloc[confirm_idx]["timestamp"]
            )

    return df


# =========================================================
# MARKET STRUCTURE / BIAS
# =========================================================

def calculate_bias(df):

    swings = add_swings(df)

    swing_highs = []
    swing_lows = []

    output = []

    for _, row in swings.iterrows():

        timestamp = row["timestamp"]

        confirmed_highs = [
            x for x in swing_highs
            if x["confirmed_at"] <= row["close_time"]
        ]

        confirmed_lows = [
            x for x in swing_lows
            if x["confirmed_at"] <= row["close_time"]
        ]

        bias = "Neutral"

        if len(confirmed_highs) >= 2 and len(confirmed_lows) >= 2:

            latest_high = confirmed_highs[-1]["level"]
            previous_high = confirmed_highs[-2]["level"]

            latest_low = confirmed_lows[-1]["level"]
            previous_low = confirmed_lows[-2]["level"]

            if (
                latest_high > previous_high
                and latest_low > previous_low
            ):
                bias = "Bullish"

            elif (
                latest_high < previous_high
                and latest_low < previous_low
            ):
                bias = "Bearish"

        output.append({
            "timestamp": timestamp,
            "close_time": row["close_time"],
            "bias": bias
        })

        if row["swing_high"]:
            swing_highs.append({
                "level": row["swing_high_level"],
                "confirmed_at": row["swing_high_confirmed_at"]
            })

        if row["swing_low"]:
            swing_lows.append({
                "level": row["swing_low_level"],
                "confirmed_at": row["swing_low_confirmed_at"]
            })

    return pd.DataFrame(output)


# =========================================================
# POINT-IN-TIME BIAS
# =========================================================

def get_bias_at(structure_df, signal_close_time):

    if structure_df.empty:
        return "Neutral"

    valid = structure_df[
        structure_df["close_time"] <= signal_close_time
    ]

    if valid.empty:
        return "Neutral"

    return valid.iloc[-1]["bias"]


# =========================================================
# 15M SWINGS
# =========================================================

def get_confirmed_swings(df, signal_time):

    swings = add_swings(df)

    highs = swings[
        (swings["swing_high"])
        & (
            swings["swing_high_confirmed_at"]
            <= signal_time
        )
    ].copy()

    lows = swings[
        (swings["swing_low"])
        & (
            swings["swing_low_confirmed_at"]
            <= signal_time
        )
    ].copy()

    return highs, lows


# =========================================================
# FIND STRATEGY V1 SIGNAL
# =========================================================

def find_signal(
    stock,
    df,
    one_hour,
    four_hour,
    latest_candle_time,
    now
):

    df = df.sort_values("timestamp").reset_index(drop=True)

    # We only evaluate the latest COMPLETED 15M candle.
    row_matches = df[
        df["timestamp"] == latest_candle_time
    ]

    if row_matches.empty:
        return None

    signal_idx = row_matches.index[-1]

    signal_row = df.loc[signal_idx]

    signal_close_time = (
        signal_row["timestamp"]
        + timedelta(minutes=15)
    )

    # Need enough history
    if signal_idx < 30:
        return None

    # -----------------------------------------------------
    # HTF BIAS
    # -----------------------------------------------------

    bias_1h = get_bias_at(
        one_hour,
        signal_close_time
    )

    bias_4h = get_bias_at(
        four_hour,
        signal_close_time
    )

    if bias_1h == "Neutral" or bias_4h == "Neutral":
        return None

    # -----------------------------------------------------
    # 15M STRUCTURE
    # -----------------------------------------------------

    swings = add_swings(df)

    confirmed_highs = swings[
        (swings["swing_high"])
        &
        (
            swings["swing_high_confirmed_at"]
            <= signal_row["timestamp"]
        )
    ]

    confirmed_lows = swings[
        (swings["swing_low"])
        &
        (
            swings["swing_low_confirmed_at"]
            <= signal_row["timestamp"]
        )
    ]

    if confirmed_highs.empty and confirmed_lows.empty:
        return None

    latest_high = (
        confirmed_highs.iloc[-1]["swing_high_level"]
        if not confirmed_highs.empty
        else np.nan
    )

    latest_low = (
        confirmed_lows.iloc[-1]["swing_low_level"]
        if not confirmed_lows.empty
        else np.nan
    )

    # -----------------------------------------------------
    # SEARCH BACK FOR RECENT SWEEP
    # -----------------------------------------------------

    start_idx = max(
        0,
        signal_idx - MAX_BOS_WAIT
    )

    recent = df.iloc[start_idx:signal_idx+1]

    # -----------------------------------------------------
    # BUY
    # -----------------------------------------------------

    if bias_4h == "Bullish" and bias_1h == "Bullish":

        for sweep_idx in range(
            signal_idx - 1,
            start_idx - 1,
            -1
        ):

            sweep = df.loc[sweep_idx]

            sweep_time = sweep["timestamp"]

            prior_lows = confirmed_lows[
                confirmed_lows[
                    "swing_low_confirmed_at"
                ] < sweep_time
            ]

            if prior_lows.empty:
                continue

            sweep_level = prior_lows.iloc[-1]["swing_low_level"]

            # Liquidity sweep
            if (
                sweep["low"] < sweep_level
                and sweep["close"] > sweep_level
            ):

                sweep_extreme = sweep["low"]

                # BOS must happen AFTER sweep
                for bos_idx in range(
                    sweep_idx + 1,
                    min(
                        signal_idx + 1,
                        sweep_idx + MAX_BOS_WAIT + 1
                    )
                ):

                    bos = df.loc[bos_idx]

                    # Relevant confirmed high before BOS
                    highs_before_bos = confirmed_highs[
                        confirmed_highs[
                            "swing_high_confirmed_at"
                        ] <= bos["timestamp"]
                    ]

                    if highs_before_bos.empty:
                        continue

                    bos_level = (
                        highs_before_bos.iloc[-1]
                        ["swing_high_level"]
                    )

                    # CLOSE must break level
                    if bos["close"] > bos_level:

                        entry = bos["close"]

                        sl = (
                            sweep_extreme
                            * (1 - SL_BUFFER)
                        )

                        risk = entry - sl

                        if risk <= 0:
                            continue

                        target = entry + (
                            risk * MIN_RR
                        )

                        # Signal must be on latest completed candle
                        if (
                            bos["timestamp"]
                            != latest_candle_time
                        ):
                            continue

                        return {
                            "stock": stock,
                            "direction": "BUY",
                            "bos_time": bos["timestamp"],
                            "entry": entry,
                            "stop_loss": sl,
                            "target": target,
                            "risk": risk,
                            "sweep_time": sweep_time,
                            "sweep_extreme": sweep_extreme,
                            "bos_level": bos_level,
                            "bars_sweep_to_bos":
                                bos_idx - sweep_idx,
                            "bias_1h": bias_1h,
                            "bias_4h": bias_4h
                        }

    # -----------------------------------------------------
    # SELL
    # -----------------------------------------------------

    if bias_4h == "Bearish" and bias_1h == "Bearish":

        for sweep_idx in range(
            signal_idx - 1,
            start_idx - 1,
            -1
        ):

            sweep = df.loc[sweep_idx]

            sweep_time = sweep["timestamp"]

            prior_highs = confirmed_highs[
                confirmed_highs[
                    "swing_high_confirmed_at"
                ] < sweep_time
            ]

            if prior_highs.empty:
                continue

            sweep_level = (
                prior_highs.iloc[-1]
                ["swing_high_level"]
            )

            # Liquidity sweep
            if (
                sweep["high"] > sweep_level
                and sweep["close"] < sweep_level
            ):

                sweep_extreme = sweep["high"]

                # BOS after sweep
                for bos_idx in range(
                    sweep_idx + 1,
                    min(
                        signal_idx + 1,
                        sweep_idx + MAX_BOS_WAIT + 1
                    )
                ):

                    bos = df.loc[bos_idx]

                    lows_before_bos = confirmed_lows[
                        confirmed_lows[
                            "swing_low_confirmed_at"
                        ] <= bos["timestamp"]
                    ]

                    if lows_before_bos.empty:
                        continue

                    bos_level = (
                        lows_before_bos.iloc[-1]
                        ["swing_low_level"]
                    )

                    if bos["close"] < bos_level:

                        entry = bos["close"]

                        sl = (
                            sweep_extreme
                            * (1 + SL_BUFFER)
                        )

                        risk = sl - entry

                        if risk <= 0:
                            continue

                        target = entry - (
                            risk * MIN_RR
                        )

                        if (
                            bos["timestamp"]
                            != latest_candle_time
                        ):
                            continue

                        return {
                            "stock": stock,
                            "direction": "SELL",
                            "bos_time": bos["timestamp"],
                            "entry": entry,
                            "stop_loss": sl,
                            "target": target,
                            "risk": risk,
                            "sweep_time": sweep_time,
                            "sweep_extreme": sweep_extreme,
                            "bos_level": bos_level,
                            "bars_sweep_to_bos":
                                bos_idx - sweep_idx,
                            "bias_1h": bias_1h,
                            "bias_4h": bias_4h
                        }

    return None


# =========================================================
# TELEGRAM
# =========================================================

def send_telegram(signal):

    direction = signal["direction"]

    emoji = "🟢" if direction == "BUY" else "🔴"

    message = f"""
{emoji} STRATEGY V1 ALERT

Stock: {signal["stock"]}
Direction: {direction}

Entry: {signal["entry"]:.2f}
Stop Loss: {signal["stop_loss"]:.2f}
Target (3R): {signal["target"]:.2f}

4H Bias: {signal["bias_4h"]}
1H Bias: {signal["bias_1h"]}

Sweep: {signal["sweep_time"]}
BOS: {signal["bos_time"]}

Bars Sweep → BOS:
{signal["bars_sweep_to_bos"]}

⚠️ PAPER ALERT ONLY
Manual confirmation required.
"""

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        },
        timeout=15
    )

    response.raise_for_status()


# =========================================================
# MAIN SCANNER
# =========================================================

def main():

    now = datetime.now(IST)

    print("=" * 60)
    print("TRADING AGENT V1 - PAPER SCANNER")
    print("Time:", now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("=" * 60)

    # -----------------------------------------------------
    # MARKET HOURS
    # -----------------------------------------------------

    if now.weekday() >= 5:

        print("Market closed - weekend.")
        return

    market_open = now.replace(
        hour=9,
        minute=15,
        second=0,
        microsecond=0
    )

    market_close = now.replace(
        hour=15,
        minute=30,
        second=0,
        microsecond=0
    )

    if now < market_open or now > market_close:

        print("Outside NSE market hours.")
        return

    # -----------------------------------------------------
    # COMPLETED 15M CANDLE
    # -----------------------------------------------------

    # Candle start + 15 minutes must be <= current time.
    current_minute = now.minute

    completed_minute = (
        (current_minute // 15) * 15
    )

    latest_completed_time = now.replace(
        minute=completed_minute,
        second=0,
        microsecond=0
    ) - timedelta(minutes=15)

    # Special handling after candle boundary
    if current_minute % 15 == 0:
        latest_completed_time = now.replace(
            minute=current_minute,
            second=0,
            microsecond=0
        ) - timedelta(minutes=15)

    print(
        "Latest completed 15M candle:",
        latest_completed_time
    )

    # -----------------------------------------------------
    # INSTRUMENTS
    # -----------------------------------------------------

    instruments = get_instrument_map()

    signals = []

    # -----------------------------------------------------
    # SCAN 30 STOCKS
    # -----------------------------------------------------

    for stock in STOCKS:

        try:

            print(f"Scanning {stock}...")

            df = fetch_15m(
                stock,
                instruments[stock],
                now.date()
            )

            if df.empty:
                print("  No data")
                continue

            # Only data available up to completed candle
            df = df[
                df["timestamp"]
                <= latest_completed_time
            ].copy()

            if df.empty:
                continue

            one_hour = build_1h(df)
            four_hour = build_4h(df)

            one_hour = calculate_bias(one_hour)
            four_hour = calculate_bias(four_hour)

            signal = find_signal(
                stock,
                df,
                one_hour,
                four_hour,
                latest_completed_time,
                now
            )

            if signal:

                print(
                    f"  SIGNAL: "
                    f"{signal['direction']}"
                )

                signals.append(signal)

            else:

                print("  No signal")

        except Exception as e:

            print(
                f"  ERROR {stock}: {e}"
            )

    # -----------------------------------------------------
    # TELEGRAM
    # -----------------------------------------------------

    print("-" * 60)
    print("Signals found:", len(signals))

    for signal in signals:

        send_telegram(signal)

        print(
            "Telegram alert sent:",
            signal["stock"],
            signal["direction"]
        )

    print("=" * 60)
    print("SCAN COMPLETE")


if __name__ == "__main__":
    main()
