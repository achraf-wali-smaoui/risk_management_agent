"""
quantile_risk_agent.py
============================================================
Quantile-based risk estimation for XDTA Risk Management Agent.

This module provides:
1) ATR-based volatility proxy computation (interpretable, widely used).
2) Quantile-based volatility regime classification (level + trend).
3) Quantile regression forecasting (sklearn QuantileRegressor) of future risk proxy.
4) A clean payload interface for downstream components (RMA / Coordinator / RiskEnv).

Design notes (Q1-style):
- The agent is a *risk estimator* (measures + forecasts risk), not a trader.
- It outputs interpretable signals usable by a Risk Policy (RL) or a rule-based Risk Manager.
- Several "constants" are exposed as hyperparameters (explicit axes of research).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Literal, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

import risk_features
from data import normalize_ohlcv_df

# ============================================================
# Config objects (with backward-compatible aliases)
# ============================================================

#TODO move this declaration of classes to wandb configuration
@dataclass
class RegimeConfig:
    """
    Configuration for quantile-based volatility regimes.

    Core (recommended):
      - quantiles: (q_lo, q_hi) boundaries for volatility level classification.
      - trend_fraction: rho in [0,1], fraction of regime_window used for "short" volatility mean.

    Backward-compatible aliases (for notebooks / older scripts):
      - atr_window: alias for ATR period (moved to agent-level as atr_period).
      - trend_window: alias to set regime_window (approx) if regime_window not given.
    """
    # Core
    quantiles: Tuple[float, float] = (0.25, 0.75)
    trend_fraction: float = 0.2

    # Aliases (optional)
    atr_window: Optional[int] = None
    trend_window: Optional[int] = None  # if user passes 20, we can map it to regime_window heuristic

#TODO move this declaration of classes to wandb configuration
@dataclass
class QRConfig:
    """
    Configuration for quantile regression forecasting.

    Core (recommended):
      - horizon: forecast horizon in bars (H)
      - feature_window: rolling window length used to build lagged/statistical features
      - quantile_levels: quantiles learned by QuantileRegressor
      - alpha: L2 regularization
      - alpha_mode: "fixed" or "vol_scaled"

    Backward-compatible aliases:
      - horizon_bars: alias for horizon
      - quantile: alias for a single upper-tail quantile (e.g. 0.95)
    """
    # Core
    horizon: int = 4
    feature_window: int = 60
    quantile_levels: Tuple[float, ...] = (0.05, 0.5, 0.95)
    alpha: float = 1e-3
    alpha_mode: Literal["fixed", "vol_scaled"] = "vol_scaled"

    # Aliases (optional)
    horizon_bars: Optional[int] = None
    quantile: Optional[float] = None  # e.g. 0.95

#TODO move this declaration of classes to wandb configuration
@dataclass
class QuantileRiskOutput:
    """
    Output produced at time t (latest bar).

    atr_t: current ATR value.
    atr_z: rolling z-score of ATR (scale-invariant proxy).
    level: volatility level label (low/mid/high).
    trend: volatility trend label (up/down/flat).
    regime_key: combined label (e.g. "high_up").
    regime_factor: multiplicative factor k_regime(t) (interpretable mapping).
    q_forecasts: predicted future ATR (risk proxy) at multiple quantiles.
    q_factor: normalized quantile factor k_quantile(t) derived from upper-tail forecast.
    meta: diagnostics and configs used.
    """
    timestamp: str
    atr_t: float
    atr_z: float
    level: risk_features.LevelLabel
    trend: risk_features.TrendLabel
    regime_key: str
    regime_factor: float
    q_forecasts: Dict[str, float]
    q_factor: float
    meta: Dict[str, Any]



# ============================================================
# Main Agent
# ============================================================

class QuantileRiskAgent:
    """
    QuantileRiskAgent: computes volatility regimes + quantile regression forecasts.

    Input:
      - df_ohlcv: DataFrame with OHLCV (real data)

    Output:
      - QuantileRiskOutput (serializable, loggable, passable to other XDTA modules)
    """

    def __init__(
        self,
        *,
        atr_period: int = 14,
        regime_window: int = 252,
        regime_config: RegimeConfig = RegimeConfig(),
        qr_config: QRConfig = QRConfig(),
        regime_k_map: Optional[Dict[str, float]] = None,
        trend_multipliers: Tuple[float, float] = (1.2, 0.9),
        random_state: int = 42,
    ):
        self.random_state = int(random_state)

        # --- Backward-compat handling (RegimeConfig aliases) ---
        if regime_config.atr_window is not None:
            atr_period = int(regime_config.atr_window)

        # If user provides "trend_window" in config, treat it as a hint for regime_window
        # (Note: regime_window is a longer calibration window; trend_window is often short.
        # We keep regime_window dominant if explicitly provided.)
        if regime_config.trend_window is not None and (regime_window is None or regime_window <= 0):
            regime_window = int(max(50, regime_config.trend_window * 10))

        self.atr_period = int(atr_period)
        self.regime_window = int(regime_window)
        self.regime_config = regime_config

        # --- Backward-compat handling (QRConfig aliases) ---
        if qr_config.horizon_bars is not None:
            qr_config.horizon = int(qr_config.horizon_bars)

        if qr_config.quantile is not None:
            # User provided a single quantile (e.g., 0.95). Keep median + that quantile by default.
            q = float(qr_config.quantile)
            qr_config.quantile_levels = tuple(sorted(set((0.5, q))))

        self.qr_config = qr_config

        # Interpretable mapping (explicitly tuneable later)
        self.regime_k_map = regime_k_map or {
            "low_flat": 1.05, "low_up": 1.10, "low_down": 1.00,
            "mid_flat": 1.00, "mid_up": 1.10, "mid_down": 0.95,
            "high_flat": 0.85, "high_up": 0.75, "high_down": 0.90,
        }

        self.tau_up, self.tau_down = float(trend_multipliers[0]), float(trend_multipliers[1])

        self._models: Dict[float, QuantileRegressor] = {}
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def run(
        self,
        df_ohlcv: pd.DataFrame,
        *,
        ticker: str = "UNKNOWN",
        timeframe: str = "UNKNOWN",
        horizon_bars: Optional[int] = None,
        use_mock_quantile_forecasts: Optional[Dict[float, float]] = None,
    ) -> QuantileRiskOutput:
        """
        Run the agent on OHLCV data and return the latest risk estimates.

        horizon_bars: optional override for horizon (e.g., user chooses 3).
        use_mock_quantile_forecasts: bypass QR training (for isolated testing).
        """
        
        # df = normalize_ohlcv_df(df_ohlcv) #vISAAC: we don't need to normalize the data here, it's already normalized in the RiskEnv.
        df = df_ohlcv.copy()
        # self._validate_ohlcv(df) #vISAAC: again, already validated in the RiskEnv.

        H = int(horizon_bars) if horizon_bars is not None else int(self.qr_config.horizon)

        # --- Adaptive regime window ---
        # If the user provides fewer bars than regime_window (common in notebooks / short samples),
        # rolling quantiles would be mostly NaN. We adapt the effective window to the available history
        # while keeping a minimum size for stability.
        eff_regime_window = int(min(self.regime_window, max(50, len(df) // 2)))

        # 1) ATR (risk proxy)
        atr = risk_features.compute_atr(df, period=self.atr_period)
        atr_z = risk_features.safe_zscore(atr, window=max(30, eff_regime_window // 5))

        atr_t = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else float(np.nan)
        atr_z_t = float(atr_z.iloc[-1]) if np.isfinite(atr_z.iloc[-1]) else 0.0

        # 2) Regime classification: level + trend
        q_lo, q_hi = self.regime_config.quantiles
        bounds = risk_features._rolling_quantile_bounds(atr, eff_regime_window, q_lo, q_hi)

        q_lo_t = float(bounds["q_lo"].iloc[-1]) if np.isfinite(bounds["q_lo"].iloc[-1]) else np.nan
        q_hi_t = float(bounds["q_hi"].iloc[-1]) if np.isfinite(bounds["q_hi"].iloc[-1]) else np.nan

        level = risk_features.classify_level(atr_t, q_lo_t, q_hi_t)

        tail_vol = atr.iloc[-eff_regime_window:] if len(atr) >= eff_regime_window else atr
        trend = risk_features.classify_trend(tail_vol, self.regime_config.trend_fraction)

        regime_key = f"{level}_{trend}"

        # 3) Regime factor k_regime(t)
        k_regime = float(self.regime_k_map.get(regime_key, 1.0))
        if trend == "up":
            k_regime *= self.tau_up
        elif trend == "down":
            k_regime *= self.tau_down

        # 4) Quantile forecasts (QR)
        if use_mock_quantile_forecasts is not None:
            q_forecasts = {str(q): float(v) for q, v in use_mock_quantile_forecasts.items()}
        else:
            q_forecasts = self.fit_predict_quantiles(atr=atr, horizon=H)

        # 5) Convert forecasts -> k_quantile(t)
        k_quantile = self._quantile_factor_from_forecasts(q_forecasts=q_forecasts, atr=atr)

        ts = str(df.index[-1])

        meta = {
            "ticker": ticker,
            "timeframe": timeframe,
            "atr_period": self.atr_period,
            "regime_window": eff_regime_window,
            "regime_quantiles": self.regime_config.quantiles,
            "trend_fraction": self.regime_config.trend_fraction,
            "qr_horizon": H,
            "qr_feature_window": self.qr_config.feature_window,
            "qr_quantiles": self.qr_config.quantile_levels,
            "qr_alpha": self.qr_config.alpha,
            "qr_alpha_mode": self.qr_config.alpha_mode,
            "trend_multipliers": (self.tau_up, self.tau_down),
            "regime_k_map_size": len(self.regime_k_map),
        }

        return QuantileRiskOutput(
            timestamp=ts,
            atr_t=atr_t,
            atr_z=atr_z_t,
            level=level,
            trend=trend,
            regime_key=regime_key,
            regime_factor=k_regime,
            q_forecasts=q_forecasts,
            q_factor=float(k_quantile),
            meta=meta,
        )

    # ------------------------------------------------------------
    # Quantile regression
    # ------------------------------------------------------------
    def fit_predict_quantiles(self, *, atr: pd.Series, horizon: int) -> Dict[str, float]:
        """
        Fit quantile regressors on ATR-derived features and predict ATR_{t+horizon}.
        """
        y = atr.shift(-horizon)
        X = risk_features.build_qr_features(atr, window=self.qr_config.feature_window)

        df = pd.concat([X, y.rename("y")], axis=1).dropna()
        if len(df) < 50:
            return self._naive_quantile_forecasts(atr)

        X_mat = df.drop(columns=["y"]).values
        y_vec = df["y"].values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_mat)

        alpha = float(self.qr_config.alpha)
        if self.qr_config.alpha_mode == "vol_scaled":
            med = float(np.nanmedian(atr.values)) if np.isfinite(np.nanmedian(atr.values)) else 1.0
            alpha = alpha * max(med, 1e-6)

        x_last = X.iloc[[-1]].values
        x_last_scaled = self._scaler.transform(x_last)

        forecasts: Dict[str, float] = {}
        for q in self.qr_config.quantile_levels:
            qf = float(q)
            model = QuantileRegressor(
                quantile=qf,
                alpha=alpha,
                solver="highs",
            )
            model.fit(X_scaled, y_vec)
            self._models[qf] = model

            pred = float(model.predict(x_last_scaled)[0])
            forecasts[str(qf)] = pred

        return forecasts

    # ------------------------------------------------------------
    # Forecasts -> normalized factor
    # ------------------------------------------------------------
    def _quantile_factor_from_forecasts(self, *, q_forecasts: Dict[str, float], atr: pd.Series) -> float:
        """
        Convert quantile forecasts into a normalized factor k_quantile(t).

        Uses upper-tail forecast / typical ATR ratio => bounded inverse mapping.
        """
        qs = sorted([float(k) for k in q_forecasts.keys()])
        q_hi = qs[-1]
        pred_hi = float(q_forecasts[str(q_hi)])

        ref = float(np.nanmedian(atr.iloc[-max(50, self.qr_config.feature_window):].values))
        ref = max(ref, 1e-6)

        ratio = pred_hi / ref
        beta = 0.7
        k = 1.0 / (ratio ** beta)

        return float(np.clip(k, 0.5, 1.5))

    # ------------------------------------------------------------
    # Fallbacks + validation
    # ------------------------------------------------------------
    def _naive_quantile_forecasts(self, atr: pd.Series) -> Dict[str, float]:
        """Fallback: empirical quantiles of recent ATR."""
        tail = atr.dropna().iloc[-max(100, self.qr_config.feature_window):]
        if len(tail) < 10:
            base = float(atr.dropna().iloc[-1]) if atr.dropna().shape[0] else 1e-6
            return {str(float(q)): base for q in self.qr_config.quantile_levels}

        out: Dict[str, float] = {}
        for q in self.qr_config.quantile_levels:
            out[str(float(q))] = float(tail.quantile(float(q)))
        return out

    # @staticmethod
    # def _validate_ohlcv(df: pd.DataFrame) -> None:
    #     required = {"Open", "High", "Low", "Close", "Volume"}
    #     missing = required - set(df.columns)
    #     if missing:
    #         raise ValueError(f"OHLCV DataFrame missing columns: {missing}")
    #     if len(df) < 50:
    #         raise ValueError("Not enough rows. Provide at least ~50 bars for stable estimates.")


# ============================================================
# Minimal executable test (standalone)
# ============================================================

def _make_mock_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV (quick smoke test)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="H")

    rets = rng.normal(0, 0.002, size=n)
    vol_bump = np.sin(np.linspace(0, 12 * np.pi, n)) * 0.0015
    rets = rets + vol_bump * rng.normal(0, 1, size=n)

    price = 100 * np.exp(np.cumsum(rets))
    close = pd.Series(price, index=idx)

    high = close * (1 + np.abs(rng.normal(0, 0.0015, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0015, size=n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = rng.integers(1000, 5000, size=n)

    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


if __name__ == "__main__":
    df = _make_mock_ohlcv()

    # Backward-compatible style (like your notebook)
    regime_cfg = RegimeConfig(quantiles=(0.1, 0.9), atr_window=14, trend_window=20)
    qr_cfg = QRConfig(quantile=0.95, horizon_bars=3, alpha=0.01)

    agent = QuantileRiskAgent(
        regime_config=regime_cfg,
        qr_config=qr_cfg,
        regime_window=252,  # long calibration window (kept explicit)
        trend_multipliers=(1.2, 0.9),
    )

    out = agent.run(df, ticker="MOCK", timeframe="1H", horizon_bars=3)
    print("\n=== QuantileRiskAgent Output (latest) ===")
    print(asdict(out))
