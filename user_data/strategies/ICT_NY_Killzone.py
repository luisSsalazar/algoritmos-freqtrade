# =============================================================================
# Freqtrade Strategy: ICT New York AM Killzone (Konzerva Risk)
# Timeframe : 15m
# Concepts  : Smart Money Concepts (SMC) / Inner Circle Trader (ICT)
# =============================================================================

from datetime import datetime
import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy, RealParameter, informative
from pandas import DataFrame
from zoneinfo import ZoneInfo

class ICT_NY_Killzone(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    # ── ROI / Stoploss (disabled – handled in custom hooks) ───────────────────
    minimal_roi = {"0": 100}
    stoploss = -0.99

    # ── Risk Constants & Parameters ───────────────────────────────────────────
    _HARDCODED_STOPLOSS: float = -0.01  # 1% adverse move
    risk_reward_ratio = RealParameter(1.5, 3.5, default=2.38, space='sell', optimize=True, load=True)

    # ── Misc settings ─────────────────────────────────────────────────────────
    startup_candle_count: int = 200      
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # ── NY Killzone window (New York local time, 24-h clock) ──────────────────
    _KZ_START_HOUR: int = 8
    _KZ_START_MIN: int = 30
    _KZ_END_HOUR: int = 11
    _KZ_END_MIN: int = 30

    # ── Asian Session window (NY local time, previous evening) ───────────────
    _ASIA_START_HOUR: int = 20
    _ASIA_END_HOUR: int = 24            

    # ==========================================================================
    # INFORMATIVE – 4H EMA 200
    # ==========================================================================
    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema200"] = pta.ema(dataframe["close"], length=200)
        return dataframe

    # ==========================================================================
    # INDICATORS  (15m frame)
    # ==========================================================================
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dates_utc = pd.to_datetime(dataframe["date"], utc=True)
        dates_ny = dates_utc.dt.tz_convert("America/New_York")

        dataframe["ny_hour"] = dates_ny.dt.hour
        dataframe["ny_min"] = dates_ny.dt.minute
        dataframe["ny_date"] = dates_ny.dt.date

        # ── 1. New York Killzone flag ─────────────────────────────────────────
        in_kz = (
            (dates_ny.dt.hour > self._KZ_START_HOUR)
            | (
                (dates_ny.dt.hour == self._KZ_START_HOUR)
                & (dates_ny.dt.minute >= self._KZ_START_MIN)
            )
        ) & (
            (dates_ny.dt.hour < self._KZ_END_HOUR)
            | (
                (dates_ny.dt.hour == self._KZ_END_HOUR)
                & (dates_ny.dt.minute <= self._KZ_END_MIN)
            )
        )
        dataframe["in_killzone"] = in_kz

        # ── 2. Asian Session High / Low ───────────────────────────────────────
        is_asian = (dates_ny.dt.hour >= self._ASIA_START_HOUR) & (
            dates_ny.dt.hour < self._ASIA_END_HOUR
        )
        dataframe["_is_asian"] = is_asian

        session_start = is_asian & (~is_asian.shift(1).fillna(False))
        dataframe["_session_id"] = session_start.cumsum()

        asian_only = dataframe.loc[is_asian, ["high", "low", "_session_id"]].copy()
        asian_grp = asian_only.groupby("_session_id")
        dataframe.loc[is_asian, "_asian_sess_high"] = asian_grp["high"].transform("max")
        dataframe.loc[is_asian, "_asian_sess_low"] = asian_grp["low"].transform("min")

        dataframe["asian_high"] = dataframe["_asian_sess_high"].ffill()
        dataframe["asian_low"] = dataframe["_asian_sess_low"].ffill()

        # ── 3. Liquidity Sweep detection ──────────────────────────────────────
        dataframe["swept_asian_low"] = (dataframe["low"] < dataframe["asian_low"]) & (
            dataframe["close"] > dataframe["asian_low"]
        )
        dataframe["swept_asian_high"] = (dataframe["high"] > dataframe["asian_high"]) & (
            dataframe["close"] < dataframe["asian_high"]
        )

        # ── 4. Fair Value Gap (FVG) detection  ───────────────────────────────
        high_1 = dataframe["high"].shift(2)   
        low_1  = dataframe["low"].shift(2)    
        high_3 = dataframe["high"]            
        low_3  = dataframe["low"]             

        bull_fvg_condition = low_3 > high_1
        bear_fvg_condition = high_3 < low_1

        dataframe["_raw_fvg_bull_bot"] = np.where(bull_fvg_condition, high_1, np.nan)
        dataframe["_raw_fvg_bull_top"] = np.where(bull_fvg_condition, low_3, np.nan)
        dataframe["_raw_fvg_bear_bot"] = np.where(bear_fvg_condition, high_3, np.nan)
        dataframe["_raw_fvg_bear_top"] = np.where(bear_fvg_condition, low_1, np.nan)

        dataframe["fvg_bull_bot"] = pd.Series(dataframe["_raw_fvg_bull_bot"]).ffill()
        dataframe["fvg_bull_top"] = pd.Series(dataframe["_raw_fvg_bull_top"]).ffill()
        dataframe["fvg_bear_bot"] = pd.Series(dataframe["_raw_fvg_bear_bot"]).ffill()
        dataframe["fvg_bear_top"] = pd.Series(dataframe["_raw_fvg_bear_top"]).ffill()

        # ── 5. FVG "Active" flag ──────────────────────────────────────────────
        dataframe["fvg_bull_active"] = dataframe["close"] >= dataframe["fvg_bull_bot"]
        dataframe["fvg_bear_active"] = dataframe["close"] <= dataframe["fvg_bear_top"]

        # ── 6. Drop internal helper columns ───────────────────────────────────
        _drop = [
            "_is_asian", "_session_id", "_asian_sess_high", "_asian_sess_low",
            "_raw_fvg_bull_bot", "_raw_fvg_bull_top", "_raw_fvg_bear_bot", "_raw_fvg_bear_top",
        ]
        dataframe.drop(columns=_drop, inplace=True, errors="ignore")

        return dataframe

    # ==========================================================================
    # ENTRY  SIGNALS
    # ==========================================================================
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        SWEEP_LOOKBACK = 8

        # ── LONG conditions ───────────────────────────────────────────────────
        bias_long = dataframe["close"] > dataframe["ema200_4h"]

        swept_low_recent = (
            dataframe["swept_asian_low"]
            .rolling(window=SWEEP_LOOKBACK, min_periods=1)
            .max()
            .astype(bool)
        )

        fvg_bull_valid = dataframe["fvg_bull_bot"].notna() & dataframe["fvg_bull_top"].notna()

        price_in_bull_fvg = (dataframe["low"] <= dataframe["fvg_bull_top"]) & (
            dataframe["high"] >= dataframe["fvg_bull_bot"]
        )

        enter_long_conditions = (
            bias_long
            & swept_low_recent
            & fvg_bull_valid
            & price_in_bull_fvg
            & dataframe["fvg_bull_active"]
            & dataframe["in_killzone"]
        )

        dataframe.loc[enter_long_conditions, "enter_long"] = 1
        dataframe.loc[enter_long_conditions, "enter_tag"] = "ict_ny_kz_long"

        # ── SHORT conditions ──────────────────────────────────────────────────
        bias_short = dataframe["close"] < dataframe["ema200_4h"]

        swept_high_recent = (
            dataframe["swept_asian_high"]
            .rolling(window=SWEEP_LOOKBACK, min_periods=1)
            .max()
            .astype(bool)
        )

        fvg_bear_valid = dataframe["fvg_bear_bot"].notna() & dataframe["fvg_bear_top"].notna()

        price_in_bear_fvg = (dataframe["high"] >= dataframe["fvg_bear_bot"]) & (
            dataframe["low"] <= dataframe["fvg_bear_top"]
        )

        enter_short_conditions = (
            bias_short
            & swept_high_recent
            & fvg_bear_valid
            & price_in_bear_fvg
            & dataframe["fvg_bear_active"]
            & dataframe["in_killzone"]
        )

        dataframe.loc[enter_short_conditions, "enter_short"] = 1
        dataframe.loc[enter_short_conditions, "enter_tag"] = "ict_ny_kz_short"

        return dataframe

    # ==========================================================================
    # EXIT  SIGNALS
    # ==========================================================================
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # ==========================================================================
    # ▼▼▼  KONZERVA RISK-MANAGEMENT HOOKS (STATIC R:R)  ▼▼▼
    # ==========================================================================

    def custom_stoploss(
        self, pair: str, trade, current_time: datetime,
        current_rate: float, current_profit: float, after_fill: bool, **kwargs
    ) -> float:

        # Breakeven clásico: Si ganamos al menos 1R, protegemos en breakeven (+0.1%)
        if current_profit > abs(self._HARDCODED_STOPLOSS):
            return 0.001

        return self._HARDCODED_STOPLOSS

    def custom_exit(
        self, pair: str, trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs
    ):
        target_profit = abs(self._HARDCODED_STOPLOSS) * self.risk_reward_ratio.value

        if current_profit >= target_profit:
            return "target_reached_ict"

        # End of Day Flush (16:00 EST)
        ny_time = current_time.astimezone(ZoneInfo("America/New_York"))
        if ny_time.hour >= 16:
            return "eod_session_close"

        return None

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: float, max_stake: float,
        leverage: float, entry_tag: str, side: str, **kwargs
    ) -> float:

        from freqtrade.persistence import Trade

        capital = self.wallets.get_total_stake_amount()

        trades = Trade.get_trades_proxy(is_open=False)
        last_trade = max(trades, key=lambda t: t.close_date) if trades else None

        risk_frac = 0.025  # Riesgo base 2.5%
        if last_trade and last_trade.close_profit <= 0:
            risk_frac = 0.015  # Modo recuperación 1.5%

        sl_pct = abs(self._HARDCODED_STOPLOSS)
        position_size = (capital * risk_frac) / sl_pct

        return min(position_size, max_stake)