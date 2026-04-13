# =============================================================================
# Freqtrade Strategy: London Asian Volatility Breakout
# Timeframe  : 15m
# Thesis     : Volatility is cyclical. A tight Asian session (compression)
#              leads to an explosive London Open (expansion). We trade the
#              momentum breakout confirmed by a volume surge.
# =============================================================================

from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy, RealParameter
from freqtrade.persistence import Trade
from pandas import DataFrame


class London_Asian_Breakout(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # ── Absolute Exit Control (Disable Freqtrade Defaults) ────────────────────
    use_custom_stoploss = True
    trailing_stop = False
    trailing_stop_positive = None
    trailing_stop_positive_offset = 0.0
    trailing_only_offset_is_reached = False
    minimal_roi = {"0": 100}
    stoploss = -0.99

    # ── Hyperopt Parameters ───────────────────────────────────────────────────
    # 1. Maximum allowed width of the Asian Range (e.g., 0.01 = 1% height)
    max_asian_range_pct = RealParameter(0.003, 0.020, default=0.01, space='buy', optimize=True, load=True)
    # 2. Volume surge required on the breakout candle (multiplier of 20-SMA volume)
    volume_surge_mult = RealParameter(1.2, 3.0, default=1.5, space='buy', optimize=True, load=True)
    # 3. Risk Reward Ratio
    risk_reward_ratio = RealParameter(1.0, 3.0, default=1.5, space='sell', optimize=True, load=True)

    # ── Risk Constants ────────────────────────────────────────────────────────
    _HARDCODED_STOPLOSS: float = -0.01
    startup_candle_count: int = 50

    # ── Sessions (NY Local Time) ──────────────────────────────────────────────
    _ASIA_START_HOUR: int = 20
    _ASIA_END_HOUR: int = 2   # Asian range calculation ends at London open
    _LDN_END_HOUR: int = 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dates_utc = pd.to_datetime(dataframe["date"], utc=True)
        dates_ny = dates_utc.dt.tz_convert("America/New_York")
        ny_hour = dates_ny.dt.hour

        dataframe["ny_hour"] = ny_hour

        # ── 1. Killzone & Session Flags ───────────────────────────────────────
        dataframe["in_london_kz"] = (ny_hour >= self._ASIA_END_HOUR) & (ny_hour < self._LDN_END_HOUR)
        is_asian = (ny_hour >= self._ASIA_START_HOUR) | (ny_hour < self._ASIA_END_HOUR)

        # ── 2. Asian Range Calculation (Vectorized) ───────────────────────────
        sess_start = is_asian & (~is_asian.shift(1).fillna(False))
        dataframe["_session_id"] = sess_start.cumsum()

        asian_rows = dataframe.loc[is_asian, ["high", "low", "_session_id"]]
        grp = asian_rows.groupby("_session_id")

        dataframe.loc[is_asian, "_asian_h"] = grp["high"].transform("max")
        dataframe.loc[is_asian, "_asian_l"] = grp["low"].transform("min")

        dataframe["asian_high"] = dataframe["_asian_h"].ffill()
        dataframe["asian_low"] = dataframe["_asian_l"].ffill()

        # Calculate Range Percentage: (High - Low) / Low
        dataframe["asian_range_pct"] = (dataframe["asian_high"] - dataframe["asian_low"]) / dataframe["asian_low"]

        # ── 3. Volume Baseline ────────────────────────────────────────────────
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()

        dataframe.drop(columns=["_session_id", "_asian_h", "_asian_l"], inplace=True, errors="ignore")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Conditions evaluated against Hyperopt parameters
        asian_defined = dataframe["asian_high"].notna()
        is_compressed = dataframe["asian_range_pct"] <= self.max_asian_range_pct.value
        has_volume = dataframe["volume"] > (dataframe["volume_sma"] * self.volume_surge_mult.value)

        # LONG: Broke out above Asian High
        long_cond = (
            dataframe["in_london_kz"] &
            asian_defined &
            is_compressed &
            has_volume &
            (dataframe["close"] > dataframe["asian_high"]) &
            (dataframe["open"] <= dataframe["asian_high"])  # Ensures the breakout happened THIS candle
        )
        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[long_cond, "enter_tag"] = "ldn_breakout_long"

        # SHORT: Broke down below Asian Low
        short_cond = (
            dataframe["in_london_kz"] &
            asian_defined &
            is_compressed &
            has_volume &
            (dataframe["close"] < dataframe["asian_low"]) &
            (dataframe["open"] >= dataframe["asian_low"])
        )
        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[short_cond, "enter_tag"] = "ldn_breakout_short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ==========================================================================
    # ▼▼▼  KONZERVA RISK-MANAGEMENT HOOKS  ▼▼▼
    # ==========================================================================

    def custom_stoploss(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, after_fill: bool, **kwargs
    ) -> float:
        # Breakeven at 0.8R
        if current_profit > (abs(self._HARDCODED_STOPLOSS) * 0.8):
            return 0.001
        return self._HARDCODED_STOPLOSS

    def custom_exit(
        self, pair: str, trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs
    ):
        target_profit = abs(self._HARDCODED_STOPLOSS) * self.risk_reward_ratio.value
        if current_profit >= target_profit:
            return "target_reached_breakout"

        ny_time = current_time.astimezone(ZoneInfo("America/New_York"))
        if ny_time.hour >= 11:
            return "session_timeout"

        return None

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float, max_stake: float,
        leverage: float, entry_tag: str, side: str, **kwargs
    ) -> float:
        capital = self.wallets.get_total_stake_amount()
        trades = Trade.get_trades_proxy(is_open=False)
        last_trade = None
        if trades:
            last_trade = max(trades, key=lambda t: t.close_date)

        risk_frac = 0.025
        if last_trade and last_trade.close_profit <= 0:
            risk_frac = 0.015

        position_size = (capital * risk_frac) / abs(self._HARDCODED_STOPLOSS)
        return min(position_size, max_stake)
