"""
risk_env.py (corrected)
====================================================
Risk-centered Trading Environment for XDTA / EDTA (research-grade).

This file is a reviewed + corrected version of the user's RiskEnv with:
- Clearer intent/action contract (so trades actually execute when intended).
- More realistic position accounting (partial close, scale-in, flip).
- Dense, risk-aware reward that does NOT collapse to HOLD by default.
- Optional reward decomposition + execution diagnostics in `info`.
- More comments explaining variables, methods, and logic.

Core idea
---------
At each bar t:
1) Build observation s_t using data available up to and including t (no look-ahead).
2) Get an external "trade intent" (direction + baseline_size) from your upstream modules.
3) Apply the RL action a_t = [scale_size, scale_sl, scale_tp, accept] to gate/scale intent.
4) Execute position change at an execution price (slippage/spread/fees).
5) Advance to bar t+1 and compute PnL with:
   - intrabar SL/TP check using (High, Low) of bar t+1, else mark-to-market using Close(t+1).
6) Compute reward r_t from equity change + risk/turnover/smoothness penalties.

Important contracts
-------------------
- intent_provider(t, df_slice_up_to_t) must return:
    {
      "direction": "long" | "short" | "flat",
      "baseline_size": float >= 0,  # in "units", clipped to cfg.max_position_units
      "confidence": float in [0,1]  # optional (default 0)
    }

- RL action (3D):
    a[0] size_mult  in [0,1]     -> multiplies baseline_size (already risk-scaled by RiskManager if desired)
    a[1] sl_mult    in [0.5,2.0] -> multiplies RiskManager's scale_sl (and ATR-based baseline)
    a[2] tp_mult    in [0.5,2.0] -> multiplies RiskManager's scale_tp (and ATR-based baseline)

- RiskManager gate (from intent_provider):
    accept_reject in {0,1} -> hard constraint (0 forces flat / no trade)

Rationale:
- Removing 'accept' from the RL action avoids the degenerate "never trade" equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, Literal

import numpy as np
import pandas as pd

import risk_features
from data import normalize_ohlcv_df


Direction = Literal["long", "short", "flat"]


# ============================================================
# Config dataclasses
# ============================================================

@dataclass
class ExecutionConfig:
    """
    Execution and friction model.

    fee_rate:
        Proportional transaction fee (e.g., 0.0005 = 5 bps).
        Applied to traded notional: fee_rate * |delta_units| * exec_price.

    spread_bps:
        Spread penalty in basis points applied to execution price.
        (Simple approximation of half-spread; ablate if needed.)

    slippage_atr_mult:
        Slippage = slippage_atr_mult * ATR * sign(delta_units)
        (ATR as volatility proxy; delta_units sign decides direction of slippage.)
    """
    fee_rate: float = 0.0005
    spread_bps: float = 1.0
    slippage_atr_mult: float = 0.05

    atr_window: int = 14  # ATR lookback window (no lookahead)


@dataclass
class RewardConfig:
    """
    Risk-sensitive reward weights.

    We implement a dense reward based on fractional equity change:

        r_profit = lam_pnl * (pnl_t / capital_t)

    and add penalties:

        r_dd     = -lam_dd     * max(0, dd_t - dd_{t-1})     (penalize worsening drawdown only)
        r_turn   = -lam_turn   * turnover_t                  (anti-overtrading)
        r_smooth = -lam_smooth * ||a_t - a_{t-1}||_1         (discourage jittery controls)

    Optional tiny exploration bonus (very small):
        r_bonus  = +lam_bonus  * 1[trade_executed]

    Notes:
    - Penalizing *absolute* drawdown every step tends to push the agent into HOLD forever.
      Penalizing only drawdown increases keeps risk-awareness without a permanent negative tax.
    - turnover penalty prevents the exploration bonus from causing overtrading.
    """

    lam_pnl: float = 1.0
    lam_dd: float = 1.0 # 0.5
    lam_turn: float = 1e-3 # 5e-4
    lam_smooth: float = 1e-3 # 5e-4
    lam_bonus: float = 1e-4  # keep tiny; just enough to avoid "never trade" collapse
    lam_no_exposure: float = 1e-3 # 5e-4
    lam_risk: float = 0.0 # 1e-3



@dataclass
class LookbackConfig:
    """
    Adaptive lookback (simple, defendable baseline).

    L_t = clip( round(L0 / (1 + alpha * max(0, atr_z))) , Lmin, Lmax )

    Intuition:
    - When volatility increases (atr_z > 0), shorten lookback to avoid mixing regimes.
    - When volatility is calm, use longer lookback to stabilize features.
    """
    L0: int = 60
    Lmin: int = 20
    Lmax: int = 120
    alpha: float = 0.5


@dataclass
class RiskEnvConfig:
    """
    Main environment configuration.
    """
    # Portfolio constraints
    initial_capital: float = 10_000.0
    max_position_units: float = 1.0   # in "units" (user-defined; could be fraction of max size)

    # SL/TP baselines (in ATR units)
    base_sl_atr: float = 1.5
    base_tp_atr: float = 2.0

    # Sub-configs
    exec_cfg: ExecutionConfig = field(default_factory=ExecutionConfig)
    rew_cfg: RewardConfig = field(default_factory=RewardConfig)
    lb_cfg: LookbackConfig = field(default_factory=LookbackConfig)

    # Numerical safety
    reward_clip: float = 10.0         # clip reward to stabilize PPO/SAC
    obs_clip: float = 10.0            # clip observation features to avoid exploding values

    # Observation shape
    obs_lookback_max: int = 60         # L_max used to make fixed-size obs
    min_size_mult: float = 0.5   # minimum size multiplier when trade is accepted
# ============================================================
# Helper functions
# ============================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _sign_dir(direction: Direction) -> int:
    if direction == "long":
        return 1
    if direction == "short":
        return -1
    return 0


"""
@deprecated
You have another implementation in risk_features.py
"""
# def _compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
#     """Average True Range (ATR) from OHLC (no lookahead)."""
    
#     high = df["High"].astype(float)
#     low = df["Low"].astype(float)
#     close = df["Close"].astype(float)

#     prev_close = close.shift(1)
#     tr1 = (high - low).abs()
#     tr2 = (high - prev_close).abs()
#     tr3 = (low - prev_close).abs()

#     tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#     atr = tr.rolling(window=window, min_periods=window).mean()
#     return atr

"""
@deprecated
You have another implementation in risk_features.py
"""
# def _zscore(x: pd.Series, window: int = 252) -> pd.Series:
#     """Rolling z-score (robust baseline)."""
#     mu = x.rolling(window=window, min_periods=max(10, window // 5)).mean()
#     sd = x.rolling(window=window, min_periods=max(10, window // 5)).std(ddof=0)
#     z = (x - mu) / (sd.replace(0.0, np.nan))
#     return z.fillna(0.0)


# ============================================================
# Main environment
# ============================================================

class RiskEnv:
    """
    Risk-centered trading environment.

    See module docstring for the contract and semantics.
    """

    def __init__(
        self,
        df_ohlcv: pd.DataFrame,
        tickers: list[str],
        cfg: Optional[RiskEnvConfig] = None,
        decision_provider: Optional[Callable[[int, str, pd.DataFrame], Dict[str, Any]]] = None,
        risk_feature_provider: Optional[Callable[[int, str, pd.DataFrame], Dict[str, Any]]] = None,
        intent_provider: Optional[Callable[[int, str, pd.DataFrame], Dict[str, Any]]] = None,
    ) -> None:
        self.cfg = cfg or RiskEnvConfig()
        
        self.tickers = tickers

        # Copy and validate price data
        self.df = df_ohlcv.copy()
        # self._validate_df(self.df) #vISAAC: We have passed previous data controls... not necessary here

        # Precompute ATR and its z-score (baseline volatility proxy)
        # vISAAC: some of this code is duplicated in the QuantileRiskAgent, we should refactor it.
        # vISAAC: you had different windows for the ATR and the z-score, I have unified them. Check please.
        self.atr = {}
        self.atr_z = {}
        for ticker in tickers:
            self.atr[ticker] = risk_features.compute_atr(self.df.loc[ticker], period=int(self.cfg.exec_cfg.atr_window))
            self.atr_z[ticker] = risk_features.safe_zscore(self.atr[ticker], window=int(self.cfg.exec_cfg.atr_window))
            # self.atr[ticker] = _compute_atr(self.df.loc[ticker], window=int(self.cfg.exec_cfg.atr_window))
            # self.atr[ticker] = self.atr[ticker].bfill().fillna(0.0)  # avoid NaNs after warmup
            # self.atr_z[ticker] = _zscore(self.atr[ticker], window=252)

        # External providers (plug-ins)
        self.decision_provider = decision_provider
        self.risk_feature_provider = risk_feature_provider
        self.intent_provider = intent_provider

        # last intent for wrapper / diagnostics
        self.last_intent = None

        # per-state shared context: we build decision data once for the current t
        self._ctx_t: Optional[int] = None
        self._ctx_rf: Optional[Dict[str, Any]] = None
        self._ctx_intent: Optional[Dict[str, Any]] = None

        # reward-shaping state
        self.last_rm_k_final: float = 1.0
        self.last_warn_kfinal: float = 1.0
        self.last_veto_kfinal: float = 1.5
        self.last_exposure_held: float = 0.0
        self.last_accept_rm: int = 0

        # -------------------------
        # Internal state variables
        # -------------------------
        self.t: int = 0
        self.done: bool = False
        self.ticker: str = tickers[0]

        # Portfolio state
        self.capital: float = float(self.cfg.initial_capital)       # current equity (cash + PnL)
        self.peak_capital: float = float(self.cfg.initial_capital)  # historical peak equity (for drawdown)

        # Position state (signed units)
        self.position_dir: Direction = "flat"  # "long" | "short" | "flat" (redundant but convenient)
        self.position_units: float = 0.0       # signed units: >0 long, <0 short, 0 flat
        self.entry_price: float = np.nan       # average entry price of current open position

        # Risk controls (stop-loss / take-profit)
        self.sl_price: float = np.nan
        self.tp_price: float = np.nan

        # Previous action and previous drawdown (needed for smoothness + Δdrawdown penalty)
        self.prev_action = np.array([1.0, 1.0, 1.0], dtype=float)
        self.prev_drawdown: float = 0.0

        # Observation shape (fixed-size)
        self.L_max = int(self.cfg.obs_lookback_max)   # max lookback used for returns tail
        # obs = [returns_tail (L_max)] + [core (8)] + [pos_onehot (3)] => L_max + 11
        self.obs_dim = self.L_max + 8 + 3

    # -------------------------
    # Validation and reset
    # -------------------------

    # vISAAC: take care with AI-code, why 300 rows?
    # def _validate_df(self, df: pd.DataFrame) -> None:
    #     required = {"Open", "High", "Low", "Close", "Volume"}
    #     if not required.issubset(set(df.columns)):
    #         raise ValueError(f"df_ohlcv must contain columns {required}, got {set(df.columns)}")
    #     if len(df) < 300:
    #         raise ValueError("df_ohlcv too small; provide more history (>=300 rows recommended).")
    #     df.sort_index(inplace=True)

    def reset(self, start_index: int = 0, idx_ticker: int = 0) -> np.ndarray:
        """
        Reset environment state.

        start_index:
            initial time index to start episodes (avoid warmup NaNs and allow lookback features).
        """
        self.t = int(start_index)
        self.done = False
        self.ticker = self.tickers[idx_ticker]

        # Reset equity and risk trackers
        self.capital = float(self.cfg.initial_capital)
        self.peak_capital = float(self.cfg.initial_capital)
        self.prev_drawdown = 0.0

        # Reset position
        self.position_dir = "flat"
        self.position_units = 0.0
        self.entry_price = np.nan
        self.sl_price = np.nan
        self.tp_price = np.nan

        # Reset last action (neutral)
        self.prev_action = np.array([1.0, 1.0, 1.0], dtype=float)

        # Clear per-state shared context
        self._ctx_t = None
        self._ctx_rf = None
        self._ctx_intent = None

        # Reset reward-shaping state
        self.last_intent = None
        self.last_rm_k_final = 1.0
        self.last_warn_kfinal = 1.0
        self.last_veto_kfinal = 1.5
        self.last_exposure_held = 0.0
        self.last_accept_rm = 0

        return self._get_obs()

    # -------------------------
    # Observation construction
    # -------------------------

    def _adaptive_lookback(self, atr_z_t: float) -> int:
        """Compute adaptive lookback L_t from atr_z (no lookahead)."""
        lb = self.cfg.lb_cfg
        L = int(round(lb.L0 / (1.0 + lb.alpha * max(0.0, float(atr_z_t)))))
        return int(_clamp(L, lb.Lmin, lb.Lmax))
        
    def _default_risk_features(self) -> Dict[str, Any]:
        atr_z_t = float(self.atr_z[self.ticker].iloc[self.t])
        return {
            "atr_z": atr_z_t,
            "k_final": 1.0,
            "regime_factor": 1.0,
            "q_factor": 1.0,
            "warn_kfinal": 1.0,
            "veto_kfinal": 1.5,
            "level_id": 1,
            "trend_id": 1,
        }

    def _default_intent(self) -> Dict[str, Any]:
        return {
            "direction": "flat",
            "baseline_size": 0.0,
            "confidence": 0.0,
            "accept_reject": 0,
            "scale_sl": 1.0,
            "scale_tp": 1.0,
            "strategy_type": "directional",
        }

    def _invalidate_step_context(self) -> None:
        self._ctx_t = None
        self._ctx_rf = None
        self._ctx_intent = None

    def _ensure_step_context(self) -> None:
        if self._ctx_t == self.t and self._ctx_rf is not None and self._ctx_intent is not None:
            return

        rf = self._default_risk_features()
        intent = self._default_intent()

        if self.decision_provider is not None:
            dec = self.decision_provider(self.t, self.ticker, self.df) or {}
            ext_rf = dict(dec.get("risk_features", {}) or {})
            ext_intent = dict(dec.get("intent", {}) or {})
        else:
            ext_rf = {}
            ext_intent = {}

            if self.risk_feature_provider is not None:
                ext = self.risk_feature_provider(self.t, self.ticker, self.df) or {}
                ext_rf = dict(ext)

            if self.intent_provider is not None:
                raw = self.intent_provider(self.t, self.ticker, self.df) or {}
                ext_intent = dict(raw)

        # backward-compat alias
        if "quantile_factor" in ext_rf and "q_factor" not in ext_rf:
            ext_rf["q_factor"] = ext_rf["quantile_factor"]

        rf.update(ext_rf)
        intent.update(ext_intent)

        # sanitize rf
        for k in ["atr_z", "k_final", "regime_factor", "q_factor"]:
            rf[k] = float(rf.get(k, 1.0))

        rf["warn_kfinal"] = float(rf.get("warn_kfinal", 1.0))
        rf["veto_kfinal"] = float(rf.get("veto_kfinal", max(rf["warn_kfinal"] + 1e-6, 1.5)))

        # sanitize intent
        direction = str(intent.get("direction", "flat")).lower().strip()
        if direction not in ("long", "short", "flat"):
            direction = "flat"

        baseline_size = float(intent.get("baseline_size", 0.0))
        baseline_size = _clamp(baseline_size, 0.0, self.cfg.max_position_units)

        confidence = float(intent.get("confidence", 0.0))
        confidence = _clamp(confidence, 0.0, 1.0)

        accept_reject = int(intent.get("accept_reject", intent.get("accept", 1)))
        accept_reject = 1 if accept_reject else 0

        scale_sl = float(intent.get("scale_sl", 1.0))
        scale_tp = float(intent.get("scale_tp", 1.0))
        scale_sl = _clamp(scale_sl, 0.1, 3.0)
        scale_tp = _clamp(scale_tp, 0.1, 3.0)

        strategy_type = str(intent.get("strategy_type", "directional")).lower().strip()
        if strategy_type not in ("directional", "volatility"):
            strategy_type = "directional"

        self._ctx_t = int(self.t)
        self._ctx_rf = rf
        self._ctx_intent = {
            "direction": direction,
            "baseline_size": baseline_size,
            "confidence": confidence,
            "accept_reject": accept_reject,
            "scale_sl": scale_sl,
            "scale_tp": scale_tp,
            "strategy_type": strategy_type,
        }


    def _get_risk_features(self) -> Dict[str, Any]:
        self._ensure_step_context()
        return dict(self._ctx_rf)

    def _get_intent(self) -> Dict[str, Any]:
        self._ensure_step_context()
        return dict(self._ctx_intent)

    def _get_obs(self) -> np.ndarray:
        """Construct a fixed-size observation vector (SB3-friendly)."""
        rf = self._get_risk_features()
        atr_z_t = float(rf["atr_z"])

        # Adaptive lookback (bounded)
        L_adapt = int(self._adaptive_lookback(atr_z_t))
        L = min(max(1, L_adapt), self.L_max)

        # Log returns up to time t
        close = self.df.loc[self.ticker, "Close"].astype(float)
        r = np.log(close / close.shift(1)).fillna(0.0)

        # Tail returns of length L, then pad/truncate to L_max
        r_tail = r.iloc[max(0, self.t - L + 1): self.t + 1].to_numpy(dtype=float)

        if len(r_tail) == 0:
            r_tail = np.zeros(self.L_max, dtype=float)
        elif len(r_tail) < self.L_max:
            pad_len = self.L_max - len(r_tail)
            pad_value = r_tail[0]   # repeat earliest available return instead of fake zeros
            pad = np.full(pad_len, pad_value, dtype=float)
            r_tail = np.concatenate([pad, r_tail])
        else:
            r_tail = r_tail[-self.L_max:]

        # Position encoding (one-hot)
        pos_onehot = np.array([
            1.0 if self.position_dir == "flat" else 0.0,
            1.0 if self.position_dir == "long" else 0.0,
            1.0 if self.position_dir == "short" else 0.0,
        ], dtype=float)

        # Exposure and unrealized PnL as fractions of equity
        price_t = float(self.df.loc[self.ticker, "Close"].iloc[self.t])
        exposure = (abs(self.position_units) * price_t) / max(self.capital, 1e-9)

        unreal_pnl = 0.0
        if self.position_dir != "flat" and np.isfinite(self.entry_price):
            unreal_pnl = self.position_units * (price_t - float(self.entry_price))

        unreal_pnl_frac = unreal_pnl / max(self.capital, 1e-9)
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-9)

        L_norm = float(L) / float(self.L_max)

        # "core" features: keep small count and stable scales
        core = np.array([
            atr_z_t,
            float(rf["k_final"]),
            float(rf["regime_factor"]),
            float(rf["q_factor"]),
            exposure,
            unreal_pnl_frac,
            drawdown,
            L_norm,   # normalized adaptive lookback : Now L stays informative, but no longer .. much cleaner for PPO
        ], dtype=float)

        obs = np.concatenate([r_tail, core, pos_onehot], axis=0)

        # Safety check
        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(f"Obs dim mismatch: {obs.shape[0]} vs {self.obs_dim}")

        # Replace NaN/Inf with zeros before clipping
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.cfg.obs_clip, neginf=-self.cfg.obs_clip)
        
        # Clip to avoid exploding gradients
        obs = np.clip(obs, -self.cfg.obs_clip, self.cfg.obs_clip)
        
        # Final safety check: ensure no NaN/Inf remain
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Observation contains NaN/Inf after sanitization: {obs}")
        
        return obs.astype(np.float32)

    # -------------------------
    # Step mechanics
    # -------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advance one step.

        Returns:
            obs_{t+1}, reward_t, done, info
        """
        if self.done:
            return self._get_obs(), 0.0, True, {"msg": "env_done"}

        # Parse and clamp the RL action
        a = self._parse_action(action)

        # External signals at time t (no lookahead)
        rf = self._get_risk_features()
        intent = self._get_intent()
        # Store last used intent so RiskPolicy can read strategy_type
        self.last_intent = intent

        self.last_rm_k_final = float(rf.get("k_final", 1.0))
        self.last_warn_kfinal = float(rf.get("warn_kfinal", 1.0))
        self.last_veto_kfinal = float(rf.get("veto_kfinal", max(self.last_warn_kfinal + 1e-6, 1.5)))

        # -------------------------
        # 1) Apply gating and sizing
        # -------------------------
        accept_rm = int(intent.get("accept_reject", 1))
        direction: Direction = intent["direction"]
        baseline_size = float(intent["baseline_size"])

        # RiskManager scales for SL/TP (RL multiplies these)
        scale_sl_rm = float(intent.get("scale_sl", 1.0))
        scale_tp_rm = float(intent.get("scale_tp", 1.0))

        # If RiskManager rejects OR no intent, target is flat
        if accept_rm == 0 or direction == "flat" or baseline_size <= 0.0:
            target_dir: Direction = "flat"
            target_units = 0.0
        else:
            target_dir = direction
            # -------------------------------------------------
            # Enforce minimum exposure when trade is accepted
            # -------------------------------------------------
            MIN_SIZE_MULT = float(self.cfg.min_size_mult)  # e.g. 0.1

            size_mult = max(float(a[0]), MIN_SIZE_MULT)
            scaled_units = float(baseline_size * size_mult)

            scaled_units = _clamp(
                scaled_units,
                0.0,
                self.cfg.max_position_units,
            )

            target_units = scaled_units * _sign_dir(target_dir)


        # -------------------------
        # 2) Execution: turnover, costs, and realized PnL on position change
        # -------------------------
        price_t = float(self.df.loc[self.ticker, "Close"].iloc[self.t])
        atr_t = float(self.atr[self.ticker].iloc[self.t])

        # Exposure held during [t, t+1] after execution at t.
        # This is the right quantity for the risk penalty.
        capital_for_exposure = float(self.capital)
        self.last_exposure_held = float(
            (abs(target_units) * price_t) / max(capital_for_exposure, 1e-9)
        )

        pos_prev = float(self.position_units)
        entry_prev = float(self.entry_price) if np.isfinite(self.entry_price) else np.nan

        delta_units = float(target_units - pos_prev)        # signed change in units
        turnover = abs(delta_units)                         # amount traded (units)

        # Effective execution price includes spread + slippage
        exec_price = self._execution_price(price_t, atr_t, delta_units)

        # Transaction fee proportional to traded notional
        fee = self.cfg.exec_cfg.fee_rate * abs(delta_units) * exec_price

        realized_pnl = 0.0
        opened_new_position = False
        flipped = False

        # If we change position, compute realized pnl properly and update entry price
        if abs(delta_units) > 1e-12:
            # Case A: previously flat -> open new
            if abs(pos_prev) <= 1e-12:
                opened_new_position = abs(target_units) > 1e-12
                self.entry_price = exec_price if opened_new_position else np.nan

            # Case B: previously non-flat
            else:
                # If target is flat: close all at exec_price
                if abs(target_units) <= 1e-12:
                    realized_pnl += pos_prev * (exec_price - entry_prev)
                    self.entry_price = np.nan

                # If target direction is opposite -> flip:
                elif np.sign(target_units) != np.sign(pos_prev):
                    # Close previous fully, then open new at exec_price
                    realized_pnl += pos_prev * (exec_price - entry_prev)
                    self.entry_price = exec_price
                    opened_new_position = True
                    flipped = True

                # Same direction: scale in / scale out
                else:
                    # Scaling out (reducing magnitude): realize pnl on closed part
                    if abs(target_units) < abs(pos_prev):
                        closed_signed = pos_prev - target_units  # same sign as pos_prev
                        realized_pnl += closed_signed * (exec_price - entry_prev)
                        # Entry price remains the same for remaining position
                        self.entry_price = entry_prev

                    # Scaling in (increasing magnitude): update average entry price
                    else:
                        added_signed = target_units - pos_prev  # same sign as pos_prev
                        # Weighted average entry
                        new_abs = abs(target_units)
                        prev_abs = abs(pos_prev)
                        add_abs = abs(added_signed)
                        # entry price update using absolute sizes
                        self.entry_price = float((prev_abs * entry_prev + add_abs * exec_price) / max(new_abs, 1e-12))
                        # SL/TP can be recomputed (optional) — we do it for consistency
                        opened_new_position = True

            # Update position_units to target
            self.position_units = float(target_units)

            # Update position_dir
            if abs(self.position_units) <= 1e-12:
                self.position_dir = "flat"
            else:
                self.position_dir = "long" if self.position_units > 0 else "short"

            # Update SL/TP if we have a non-flat position and we opened/flipped/changed size
            if self.position_dir != "flat" and (opened_new_position or flipped):
                self._set_sl_tp(
                    entry=float(self.entry_price),
                    atr=atr_t,
                    scale_sl=(scale_sl_rm * a[1]),
                    scale_tp=(scale_tp_rm * a[2]),
                    direction=self.position_dir,
                )
            elif self.position_dir == "flat":
                self.sl_price = np.nan
                self.tp_price = np.nan

        # -------------------------
        # 3) Advance to t+1 and compute step PnL (SL/TP intrabar else MTM)
        # -------------------------
        t_next = self.t + 1
        df_ticker = self.df.loc[self.ticker]
        if t_next >= len(df_ticker) - 1:
            self.done = True

        info_event = "hold"
        step_pnl = 0.0

        if self.position_dir != "flat" and not self.done:
            high = float(self.df.loc[self.ticker, "High"].iloc[t_next])
            low = float(self.df.loc[self.ticker, "Low"].iloc[t_next])
            close_next = float(self.df.loc[self.ticker, "Close"].iloc[t_next])

            hit_sl, hit_tp = self._check_sl_tp(high, low)

            if hit_sl or hit_tp:
                exit_price = float(self.sl_price if hit_sl else self.tp_price)

                step_pnl = self.position_units * (exit_price - float(self.entry_price))
                info_event = "stop_loss" if hit_sl else "take_profit"

                # Close position
                self.position_dir = "flat"
                self.position_units = 0.0
                self.entry_price = np.nan
                self.sl_price = np.nan
                self.tp_price = np.nan
            else:
                # Mark-to-market close-to-close PnL
                step_pnl = self.position_units * (close_next - price_t)
                info_event = "mark_to_market"

        # Total pnl includes realized pnl from execution + market movement - fees
        pnl_t = realized_pnl + step_pnl - fee

        # Save equity before applying pnl for stable reward scaling
        capital_prev = float(self.capital)

        # Update capital/equity
        self.capital += pnl_t

        # Update peak and drawdown
        self.peak_capital = max(self.peak_capital, self.capital)
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-9)

        # -------------------------
        # 4) Reward (dense + risk-aware)
        # -------------------------
        
        # Store accept flag for reward shaping
        self.last_accept_rm = int(accept_rm)

        reward, r_parts = self._reward(
            pnl=pnl_t,
            drawdown=float(drawdown),
            turnover=float(turnover),
            a=a,
            a_prev=self.prev_action,
            capital_prev=capital_prev,
        )

        self.prev_action = a.copy()

        # Move time forward
        self.t = t_next
        obs_next = self._get_obs()

        # Termination if bankrupt
        if self.capital <= 0.0:
            self.done = True
            info_event = "bankrupt"

        # Info dict: debugging and training diagnostics
        info = {
            "t": self.t,
            "event": info_event,

            # Execution diagnostics
            "accept_reject": int(accept_rm),
            "scale_sl_rm": float(scale_sl_rm),
            "scale_tp_rm": float(scale_tp_rm),
            "size_mult_rl": float(a[0]),
            "sl_mult_rl": float(a[1]),
            "tp_mult_rl": float(a[2]),
            "intent_direction": str(direction),
            "intent_baseline_size": float(baseline_size),
            "target_units": float(target_units),
            "delta_units": float(delta_units),
            "turnover": float(turnover),
            "exec_price": float(exec_price),
            "fee": float(fee),

            # PnL/equity
            "realized_pnl": float(realized_pnl),
            "step_pnl": float(step_pnl),
            "pnl": float(pnl_t),
            "capital": float(self.capital),
            "drawdown": float(drawdown),

            # Risk features (from provider or baseline)
            "k_final": float(rf.get("k_final", 1.0)),
            "warn_kfinal": float(rf.get("warn_kfinal", 1.0)),
            "veto_kfinal": float(rf.get("veto_kfinal", 1.5)),
            "regime_factor": float(rf.get("regime_factor", 1.0)),
            "q_factor": float(rf.get("q_factor", 1.0)),
            "atr_z": float(rf.get("atr_z", 0.0)),
            "exposure_held": float(self.last_exposure_held),

            # Reward decomposition (super useful to debug HOLD collapse)
            "reward_parts": r_parts,

            "direction": direction,
        }

        return obs_next, float(reward), bool(self.done), info

    # -------------------------
    # Internal mechanics
    # -------------------------

    def _parse_action(self, action: np.ndarray) -> np.ndarray:
        """Parse and clamp RL action into valid bounds.

        New (Option B) action semantics:
            a = [size_mult, sl_mult, tp_mult]

        Notes:
        - The accept/reject gate is NOT learned. It is provided by RiskManager via intent["accept_reject"].
        - size_mult is in [0,1] and scales the upstream baseline_size.
        - sl_mult and tp_mult are in [0.5,2.0] and multiply the RiskManager envelope (scale_sl/scale_tp).
        """
        a = np.array(action, dtype=float).reshape(-1)
        if a.shape[0] != 3:
            raise ValueError("Action must have 3 components: [size_mult, sl_mult, tp_mult]")

        size_mult = _clamp(float(a[0]), 0.0, 1.0)
        sl_mult = _clamp(float(a[1]), 0.5, 2.0)
        tp_mult = _clamp(float(a[2]), 0.5, 2.0)

        return np.array([size_mult, sl_mult, tp_mult], dtype=float)

    def _execution_price(self, mid_price: float, atr_t: float, delta_units: float) -> float:
        """
        Compute effective execution price:
            P_exec = P_mid + spread + slippage

        spread:
            (spread_bps / 1e4) * mid_price

        slippage:
            slippage_atr_mult * ATR * sign(delta_units)
            (trade direction affects execution; positive delta pushes price up, etc.)
        """
        cfg = self.cfg.exec_cfg
        spread = (cfg.spread_bps / 10_000.0) * mid_price
        slip = cfg.slippage_atr_mult * atr_t * np.sign(delta_units)
        return float(mid_price + spread + slip)

    def _set_sl_tp(self, entry: float, atr: float, scale_sl: float, scale_tp: float, direction: Direction) -> None:
        """Set SL/TP around entry based on ATR distances and action scalings."""
        sl_dist = self.cfg.base_sl_atr * atr * float(scale_sl)
        tp_dist = self.cfg.base_tp_atr * atr * float(scale_tp)

        if direction == "long":
            self.sl_price = float(entry - sl_dist)
            self.tp_price = float(entry + tp_dist)
        elif direction == "short":
            self.sl_price = float(entry + sl_dist)
            self.tp_price = float(entry - tp_dist)
        else:
            self.sl_price = np.nan
            self.tp_price = np.nan

    def _check_sl_tp(self, high: float, low: float) -> Tuple[bool, bool]:
        """
        Determine if SL or TP is hit in the next bar.
        Using High/Low is a standard bar-based approximation.

        If both SL and TP hit in the same bar, we assume SL triggers first (conservative).
        """
        if self.position_dir == "flat":
            return False, False

        sl = float(self.sl_price)
        tp = float(self.tp_price)

        if self.position_dir == "long":
            hit_sl = low <= sl
            hit_tp = high >= tp
        else:  # short
            hit_sl = high >= sl
            hit_tp = low <= tp

        if hit_sl and hit_tp:
            hit_tp = False

        return bool(hit_sl), bool(hit_tp)
    #  Accessor for RiskPolicy to read the last strategy intent
    def get_last_intent(self):
        return self.last_intent

    def _reward(
        self,
        pnl: float,
        drawdown: float,
        turnover: float,
        a: np.ndarray,
        a_prev: np.ndarray,
        capital_prev: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Dense, risk-aware reward.

        Returns:
            reward (float), parts (dict) for debugging
        """
        rcfg = self.cfg.rew_cfg

        # 1) Profit signal (dense): equity change fraction
        r_profit = rcfg.lam_pnl * (pnl / max(capital_prev, 1e-9))

        # 2) Drawdown penalty: only penalize worsening drawdown (Δdd >= 0)
        dd_prev = float(getattr(self, "prev_drawdown", 0.0))
        dd_inc = max(0.0, drawdown - dd_prev)
        self.prev_drawdown = float(drawdown)
        r_dd = -rcfg.lam_dd * dd_inc

        # 3) Turnover penalty (anti-overtrading)
        r_turn = -rcfg.lam_turn * float(turnover)

        # 4) Smoothness penalty (avoid oscillatory controls)
        r_smooth = -rcfg.lam_smooth * float(np.linalg.norm(a - a_prev, ord=1))

        # 5) Tiny exploration bonus if a trade was executed
        trade_executed = 1.0 if turnover > 1e-12 else 0.0
        r_bonus = rcfg.lam_bonus * trade_executed

        # 6) Penalize accepted-but-no-held-exposure
        accept_rm = int(getattr(self, "last_accept_rm", 0))

        # 7) Thresholded risk penalty: only penalize exposure above the RM warning zone.
        rm_k_final = float(getattr(self, "last_rm_k_final", 1.0))
        warn_k = float(getattr(self, "last_warn_kfinal", 1.0))
        veto_k = float(getattr(self, "last_veto_kfinal", max(warn_k + 1e-6, 1.5)))
        exposure_held = float(getattr(self, "last_exposure_held", 0.0))

        denom = max(veto_k - warn_k, 1e-6)
        risk_excess = (rm_k_final - warn_k) / denom
        risk_excess = float(np.clip(risk_excess, 0.0, 1.0))

        r_risk = -rcfg.lam_risk * risk_excess * exposure_held

        r_no_exposure = 0.0
        if accept_rm == 1 and exposure_held <= 1e-12:
            r_no_exposure = -rcfg.lam_no_exposure


        reward = (
            r_profit
            + r_dd
            + r_turn
            + r_smooth
            + r_bonus
            + r_no_exposure
            + r_risk
        )


        # Clip to improve training stability
        reward = float(np.clip(reward, -self.cfg.reward_clip, self.cfg.reward_clip))

        parts = {
            "profit": float(r_profit),
            "dd": float(r_dd),
            "turn": float(r_turn),
            "smooth": float(r_smooth),
            "bonus": float(r_bonus),
            "dd_inc": float(dd_inc),
            "no_exposure": float(r_no_exposure),
            "risk": float(r_risk),
            "risk_excess": float(risk_excess),
        }
        return reward, parts
