"""
multi_asset_models_v4 the risk agent: reward-search version
based on test_adaptation_v2.py

Main additions:
- reward preset search
- random reward search
- W&B logging of reward constants
- asset/mode/profile filters
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import argparse

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

# Project root for imports
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data import get_ohlcv_from_list
from quantile_risk_agent import QuantileRiskAgent, RegimeConfig, QRConfig
from risk_manager import TradeIntent, RiskManager, RiskManagerConfig
from risk_env import RiskEnv, RiskEnvConfig, RewardConfig
from risk_policy import RiskPolicy, RiskPolicyConfig, RiskEnvGymWrapper

load_dotenv()

# Action bounds for logging
SIZE_MULT_LOW, SIZE_MULT_HIGH = 0.0, 1.0
SL_MULT_LOW, SL_MULT_HIGH = 0.5, 2.0
TP_MULT_LOW, TP_MULT_HIGH = 0.5, 2.0


# ============================================================
# W&B utils
# ============================================================

def ensure_wandb_login() -> None:
    raw_key = os.getenv("WANDB_API_KEY", "")
    api_key = raw_key.strip().strip('"').strip("'")
    if not api_key:
        print("[wandb] WANDB_API_KEY not found in environment. Using cached local login if available.")
        return

    try:
        ok = wandb.login(key=api_key, relogin=False)
    except Exception as e:
        raise RuntimeError(
            "W&B authentication failed. Check WANDB_API_KEY and network connectivity."
        ) from e

    if ok is False:
        raise RuntimeError("W&B login returned False. Verify WANDB_API_KEY.")


def print_wandb_run_info() -> None:
    run = wandb.run
    if run is None:
        raise RuntimeError("wandb.init did not create an active run.")

    run_url = run.url if getattr(run, "url", None) else "<no-url>"
    run_mode = str(getattr(run.settings, "mode", "unknown")).lower()
    print(f"[wandb] run started: {run_url} | mode={run_mode} | id={run.id}")
    if run_mode != "online":
        print(f"[wandb][WARN] Run mode is '{run_mode}'. It may not appear in the online dashboard.")


# ============================================================
# GLOBAL CONFIG
# ============================================================

PROJECT_NAME = "multi_asset_models_v4_reward_search"
TRAIN_STEPS = 120_000
Q_HORIZON = 3
BASE_DIR = "models/multi_asset_models_v4"
os.makedirs(BASE_DIR, exist_ok=True)

ASSETS = [
    ("BTC-USD",  "1h",  "6mo"),
    ("ETH-USD",  "15m", "60d"),
    ("TSLA",     "1d",  "5y"),
    ("EURUSD=X", "1d",  "5y"),
    ("XAUT-USD", "1d",  "5y"),
]

ABLATIONS = {
    "B1": dict(use_quantiles=False, lambda_q=0.0, lambda_reg=0.0),
    "B2": dict(use_quantiles=False, lambda_q=0.0, lambda_reg=1.0),
    "B3": dict(use_quantiles=True,  lambda_q=1.0, lambda_reg=0.0),
    "B4": dict(use_quantiles=True,  lambda_q=0.5, lambda_reg=0.5),
    "B5": dict(use_quantiles=True,  lambda_q=0.5, lambda_reg=0.5),
}

SWEEP_SHORT_PROFILES = [
    dict(name="S1_base", warn_kfinal=1.15, veto_kfinal=1.35, size_curve=2.00, vol_requires_extreme=True),
    dict(name="S2_tight", warn_kfinal=1.08, veto_kfinal=1.25, size_curve=2.40, vol_requires_extreme=True),
    dict(name="S3_loose", warn_kfinal=1.22, veto_kfinal=1.45, size_curve=1.60, vol_requires_extreme=True),
    dict(name="S4_mid", warn_kfinal=1.12, veto_kfinal=1.30, size_curve=1.90, vol_requires_extreme=True),
    dict(name="S5_soft_veto", warn_kfinal=1.20, veto_kfinal=1.50, size_curve=1.80, vol_requires_extreme=False),
    dict(name="S6_risk_strict", warn_kfinal=1.05, veto_kfinal=1.20, size_curve=2.60, vol_requires_extreme=True),
    dict(name="S7_vol_friendly", warn_kfinal=1.14, veto_kfinal=1.32, size_curve=2.00, vol_requires_extreme=False),
    dict(name="S8_buffer_high", warn_kfinal=1.28, veto_kfinal=1.60, size_curve=1.50, vol_requires_extreme=True),
    dict(name="S9_relaxed", warn_kfinal=1.05,veto_kfinal=1.25,size_curve=2.0,vol_requires_extreme=False),
]
SWEEP_PROFILE_MAP = {cfg["name"]: cfg for cfg in SWEEP_SHORT_PROFILES}

# ============================================================
# REWARD SEARCH SPACE
# ============================================================

REWARD_PRESETS = {
    "R0_base": dict(
        lam_pnl=1.0,
        lam_dd=1.0,
        lam_turn=1e-3,
        lam_smooth=1e-3,
        lam_bonus=1e-4,
        lam_no_exposure=1e-3,
        lam_risk=0.0,
    ),
    "R1_soft": dict(
        lam_pnl=1.0,
        lam_dd=0.5,
        lam_turn=5e-4,
        lam_smooth=5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=0.0,
    ),
    "R2_turn_light": dict(
        lam_pnl=1.0,
        lam_dd=1.0,
        lam_turn=2e-4,
        lam_smooth=5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=0.0,
    ),
    "R3_dd_light": dict(
        lam_pnl=1.0,
        lam_dd=0.25,
        lam_turn=1e-3,
        lam_smooth=1e-3,
        lam_bonus=1e-4,
        lam_no_exposure=1e-3,
        lam_risk=0.0,
    ),
    "R4_profit_plus": dict(
        lam_pnl=1.25,
        lam_dd=0.5,
        lam_turn=5e-4,
        lam_smooth=5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=0.0,
    ),
    "R5_profit_focused": dict(
        lam_pnl=2.0,
        lam_dd=0.25,
        lam_turn=2e-4,
        lam_smooth=2e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=0.0,

    ),
    "R5_profit_focused_v2": dict(
        lam_pnl=2.0,
        lam_dd=0.25,
        lam_turn=2e-4,
        lam_smooth=2e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=0.0,

    ),
    "R6_balanced": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=1e-3,
    ),
        "R6_balanced_v2": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=1e-4,
    ),
        "R6_balanced_v3": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=3e-4,
    ),
        "R6_balanced_v4": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=1e-3,
    ),
        "R6_balanced_v5": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=5e-5,
    ),
        "R6_balanced_v6": dict(
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=2e-4,
    ),
        "R6_balanced_v7": dict( 
        lam_pnl=1.5,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=5e-4,
        lam_risk=2.5e-4,
    ),
        "R7_plus_control_light": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=2.5e-4,
        lam_smooth=2.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=5e-5,
    ),
        "R7_plus_control_light_v2": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=1.5e-4,
        lam_smooth=1.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=5e-5,
    ),
        "R7_plus_control_light_v3": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=2.5e-4,
        lam_smooth=2.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=0.0,
    ),
        "R7_plus_control_light_v4": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=1.5e-4,
        lam_smooth=1.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=0,
    ),
        "R7_plus_control_light_v5": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=1.5e-4,
        lam_smooth=1.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=1e-5,
    ),
        "R7_plus_control_light_v6": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=1.5e-4,
        lam_smooth=1.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=2.5e-5,
    ),
        "R7_plus_control_light_v7": dict(
        lam_pnl=2.0,
        lam_dd=0.30,
        lam_turn=1.5e-4,
        lam_smooth=1.5e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=7.5e-5,
    ),
        "R8_plus_control_balanced": dict(
        lam_pnl=2.0,
        lam_dd=0.35,
        lam_turn=3e-4,
        lam_smooth=3e-4,
        lam_bonus=1e-4,
        lam_no_exposure=2e-4,
        lam_risk=1e-4,
    ),

}


def sample_random_reward_cfg(rng: np.random.Generator, idx: int) -> Dict[str, Any]:
    return dict(
        name=f"RR{idx:02d}",
        lam_pnl=float(rng.choice([1.0, 1.1, 1.25, 1.5])),
        lam_dd=float(rng.choice([0.25, 0.5, 0.75, 1.0])),
        lam_turn=float(rng.choice([1e-4, 2e-4, 5e-4, 1e-3])),
        lam_smooth=float(rng.choice([1e-4, 2e-4, 5e-4, 1e-3])),
        lam_bonus=float(rng.choice([5e-5, 1e-4, 2e-4])),
        lam_no_exposure=float(rng.choice([1e-4, 2e-4, 5e-4, 1e-3])),
        lam_risk=float(rng.choice([0.0, 1e-4, 3e-4, 1e-3])),
    )


def build_reward_candidates(args) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    if args.reward_presets.strip():
        names = [x.strip() for x in args.reward_presets.split(",") if x.strip()]
    else:
        names = ["R0_base"]

    unknown = [n for n in names if n not in REWARD_PRESETS]
    if unknown:
        raise ValueError(f"Unknown reward preset(s): {unknown}. Available: {sorted(REWARD_PRESETS.keys())}")

    for n in names:
        cfg = dict(REWARD_PRESETS[n])
        cfg["name"] = n
        candidates.append(cfg)

    if args.random_reward_search:
        rng = np.random.default_rng(args.reward_seed)
        for i in range(1, int(args.reward_trials) + 1):
            candidates.append(sample_random_reward_cfg(rng, i))

    return candidates


# ============================================================
# QUANTILE RISK AGENT
# ============================================================

regime_cfg = RegimeConfig(
    quantiles=(0.25, 0.75),
    atr_window=14,
    trend_fraction=0.2,
    trend_window=20,
)

qr_cfg = QRConfig(
    horizon_bars=Q_HORIZON,
    feature_window=60,
    quantile_levels=(0.5, 0.95),
    alpha=0.01,
    alpha_mode="vol_scaled",
)

qra = QuantileRiskAgent(regime_config=regime_cfg, qr_config=qr_cfg)


# ============================================================
# INTENT PROVIDERS (updated versions)
# ============================================================

def intent_provider_directional(t, df_):
    df_ = df_.iloc[:t+1].copy()
    if t < 120:
        return None

    close = df_["Close"].astype(float)
    high = df_["High"].astype(float)
    low = df_["Low"].astype(float)

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()

    # simple ATR proxy for normalization
    tr = (high - low).rolling(14).mean()
    atr_t = float(tr.iloc[-1]) if pd.notna(tr.iloc[-1]) and tr.iloc[-1] > 1e-9 else None
    if atr_t is None:
        return None

    # trend strength normalized by ATR
    ema_gap = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
    trend_strength = ema_gap / atr_t

    # fast EMA slope over last few bars
    if len(ema_fast) < 6:
        return None
    slope = float((ema_fast.iloc[-1] - ema_fast.iloc[-5]) / max(atr_t, 1e-9))

    # breakout confirmation over recent window
    recent_high = float(high.iloc[-20:-1].max())
    recent_low = float(low.iloc[-20:-1].min())
    price_t = float(close.iloc[-1])

    direction = None
    breakout_ok = False

    if trend_strength > 0 and slope > 0:
        direction = "long"
        breakout_ok = price_t > recent_high
    elif trend_strength < 0 and slope < 0:
        direction = "short"
        breakout_ok = price_t < recent_low
    else:
        return None

    # require minimum trend quality
    if abs(trend_strength) < 0.10:
        return None

    # if no breakout, keep only stronger trends
    if not breakout_ok and abs(trend_strength) < 0.20:
        return None

    confidence = min(
        1.0,
        0.6 * abs(trend_strength) + 0.4 * abs(slope)
    )

    baseline_size = max(0.1, min(confidence, 1.0))

    return TradeIntent(
        direction=direction,
        baseline_size=float(baseline_size),
        confidence=float(confidence),
        strategy_type="directional",
    )

def intent_provider_volatility(t, df_):
    df_ = df_.iloc[:t+1].copy()

    if t < 80:
        return None

    close = df_["Close"].astype(float)
    high = df_["High"].astype(float)
    low = df_["Low"].astype(float)

    # ATR proxy
    tr = (high - low).rolling(14).mean()
    atr_t = float(tr.iloc[-1]) if pd.notna(tr.iloc[-1]) and tr.iloc[-1] > 1e-9 else None
    if atr_t is None:
        return None

    # short-term realized move
    ret_1 = float(close.iloc[-1] - close.iloc[-2])
    impulse = ret_1 / atr_t

    # breakout levels
    lookback = 20
    recent_high = float(high.iloc[-lookback:-1].max())
    recent_low = float(low.iloc[-lookback:-1].min())
    price_t = float(close.iloc[-1])

    # compression / expansion proxy
    recent_range = float((high.iloc[-10:] - low.iloc[-10:]).mean())
    older_range = float((high.iloc[-30:-10] - low.iloc[-30:-10]).mean())
    if older_range <= 1e-9:
        return None

    expansion_ratio = recent_range / older_range

    direction = None

    # require real breakout + enough impulse
    if price_t > recent_high and impulse > 0.5:
        direction = "long"
    elif price_t < recent_low and impulse < -0.5:
        direction = "short"
    else:
        return None

    # prefer expansion regimes for volatility trades
    if expansion_ratio < 1.00:
        return None

    confidence = min(
        1.0,
        0.7 * abs(impulse) + 0.3 * min(expansion_ratio, 2.0) / 2.0
    )

    baseline_size = max(0.1, min(confidence, 1.0))

    return TradeIntent(
        direction=direction,
        baseline_size=float(baseline_size),
        confidence=float(confidence),
        strategy_type="volatility",
    )

# ============================================================
# RM diagnostics utils
# ============================================================

def new_rm_trace_state() -> Dict[str, Any]:
    return {
        "steps": 0,
        "accepts": 0,
        "k_values": [],
        "veto_counts": {},
    }


def snapshot_rm_trace_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "steps": int(state.get("steps", 0)),
        "accepts": int(state.get("accepts", 0)),
        "k_len": len(state.get("k_values", [])),
        "veto_counts": dict(state.get("veto_counts", {})),
    }


def rm_trace_delta(state: Dict[str, Any], snap: Dict[str, Any]) -> Dict[str, Any]:
    steps = int(state.get("steps", 0)) - int(snap.get("steps", 0))
    accepts = int(state.get("accepts", 0)) - int(snap.get("accepts", 0))
    start_idx = int(snap.get("k_len", 0))
    all_k_values = list(state.get("k_values", []))
    new_k_values = all_k_values[start_idx:]

    if new_k_values:
        k_mean = float(np.mean(new_k_values))
        k_min = float(np.min(new_k_values))
        k_max = float(np.max(new_k_values))
    else:
        k_mean = 0.0
        k_min = 0.0
        k_max = 0.0

    cur_veto = dict(state.get("veto_counts", {}))
    old_veto = dict(snap.get("veto_counts", {}))
    keys = set(cur_veto.keys()) | set(old_veto.keys())
    delta_veto = {
        k: int(cur_veto.get(k, 0)) - int(old_veto.get(k, 0))
        for k in sorted(keys)
        if int(cur_veto.get(k, 0)) - int(old_veto.get(k, 0)) > 0
    }

    return {
        "rm_steps": max(0, steps),
        "rm_accepts": max(0, accepts),
        "rm_accept_rate": float(accepts / max(steps, 1)),
        "rm_mean_k_final": k_mean,
        "rm_k_final_min": float(k_min),
        "rm_k_final_max": float(k_max),
        "rm_veto_counts": delta_veto,
    }


def update_rm_trace_state(state: Dict[str, Any], rm_out: Any) -> None:
    state["steps"] = int(state.get("steps", 0)) + 1
    accept = int(getattr(rm_out.action, "accept_reject", 0))
    state["accepts"] = int(state.get("accepts", 0)) + accept

    k_final = float(getattr(rm_out, "k_final", 1.0))
    k_values = list(state.get("k_values", []))
    k_values.append(k_final)
    state["k_values"] = k_values

    veto_counts = dict(state.get("veto_counts", {}))
    reasons = list(getattr(rm_out, "explanation", {}).get("veto_reasons", []))
    if reasons:
        for r in reasons:
            key = str(r)
            veto_counts[key] = int(veto_counts.get(key, 0)) + 1
    else:
        veto_counts["none"] = int(veto_counts.get("none", 0)) + 1
    state["veto_counts"] = veto_counts


def sanitize_metric_key(raw: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in str(raw))
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else "unknown"


# ============================================================
# Providers bound to env
# ============================================================

def _get_risk_out(
    t,
    df_,
    cfg_quantiles,
    qra_cache: Optional[Dict[int, Any]] = None,
):
    if cfg_quantiles:
        key = int(t)
        out = qra_cache.get(key) if qra_cache is not None else None
        if out is None:
            out = qra.run(df_.iloc[:t+1], horizon_bars=Q_HORIZON)
            if qra_cache is not None:
                qra_cache[key] = out
        return out

    return type("RiskOut", (), {
        "q_factor": 1.0,
        "regime_factor": 1.0,
        "level": "none",
        "trend": "none",
        "meta": {},
    })()


def decision_provider(
    t,
    df_,
    cfg_quantiles,
    risk_manager,
    qra_cache: Optional[Dict[int, Any]] = None,
    rm_trace_state: Optional[Dict[str, Any]] = None,
):
    risk_out = _get_risk_out(
        t=t,
        df_=df_,
        cfg_quantiles=cfg_quantiles,
        qra_cache=qra_cache,
    )

    chosen = None
    rm_out = None

    base = intent_provider_directional(t, df_)
    if base is not None:
        rm_out = risk_manager.decide(risk_out, base)
        chosen = base

        if rm_out.action.accept_reject == 0:
            vol = intent_provider_volatility(t, df_)
            if vol is not None:
                rm_out = risk_manager.decide(risk_out, vol)
                chosen = vol
    else:
        vol = intent_provider_volatility(t, df_)
        if vol is not None:
            rm_out = risk_manager.decide(risk_out, vol)
            chosen = vol

    if rm_out is None or chosen is None:
        chosen = TradeIntent(
            direction="flat",
            baseline_size=0.0,
            confidence=0.0,
            strategy_type="directional",
        )
        rm_out = risk_manager.decide(risk_out, chosen)

    if rm_trace_state is not None:
        update_rm_trace_state(rm_trace_state, rm_out)

    return {
        "risk_features": {
            "k_final": float(rm_out.k_final),  # real RM intensity, not raw q_factor
            "regime_factor": float(getattr(risk_out, "regime_factor", 1.0)),
            "q_factor": float(getattr(risk_out, "q_factor", 1.0)),
            "warn_kfinal": float(risk_manager.cfg.warn_kfinal),
            "veto_kfinal": float(risk_manager.cfg.veto_kfinal),
            "level": getattr(risk_out, "level", "none"),
            "trend": getattr(risk_out, "trend", "none"),
        },
        "intent": {
            "direction": chosen.direction,
            "baseline_size": float(chosen.baseline_size) * float(rm_out.action.scale_size),
            "confidence": float(chosen.confidence),
            "accept_reject": int(rm_out.action.accept_reject),
            "scale_sl": float(rm_out.action.scale_sl),
            "scale_tp": float(rm_out.action.scale_tp),
            "strategy_type": chosen.strategy_type,
        },
    }


# ============================================================
# Eval utilities
# ============================================================

def run_episode(
    env: RiskEnv,
    policy: RiskPolicy,
    ticker_idx: int,
    start_index: int = 100,
    max_steps: int = 500,
    rm_trace_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wrapped = RiskEnvGymWrapper(env, policy.cfg)
    reset_out = wrapped.reset(options={"start_index": start_index, "idx_ticker": ticker_idx})
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    raw = wrapped.env
    initial_capital = float(raw.capital)
    total_reward = 0.0
    steps = 0
    turnover_sum = 0.0
    max_drawdown = 0.0
    invalid_actions = 0

    pnl_sum = 0.0
    realized_pnl_sum = 0.0
    step_pnl_sum = 0.0
    exposure_held_sum = 0.0

    reward_part_sums = {
        "profit": 0.0,
        "dd": 0.0,
        "turn": 0.0,
        "smooth": 0.0,
        "bonus": 0.0,
        "no_exposure": 0.0,
        "risk": 0.0,
        "risk_excess": 0.0,
    }


    rm_snapshot = snapshot_rm_trace_state(rm_trace_state) if rm_trace_state is not None else None

    while not raw.done and steps < max_steps:
        model_obs = np.asarray(obs, dtype=np.float32)
        norm_env = getattr(policy, "env", None)
        if norm_env is not None and hasattr(norm_env, "normalize_obs"):
            try:
                model_obs = np.asarray(norm_env.normalize_obs(model_obs[None, :])[0], dtype=np.float32)
            except Exception:
                pass

        action, _ = policy.model.predict(model_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if not (SIZE_MULT_LOW <= action[0] <= SIZE_MULT_HIGH):
            invalid_actions += 1
        if not (SL_MULT_LOW <= action[1] <= SL_MULT_HIGH):
            invalid_actions += 1
        if not (TP_MULT_LOW <= action[2] <= TP_MULT_HIGH):
            invalid_actions += 1

        step_out = wrapped.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, _, info = step_out

        if not np.isfinite(reward):
            raise RuntimeError(
                f"Non-finite reward at step {steps}: reward={reward}, info keys={list(info.keys())}"
            )

        total_reward += float(reward)
        turnover_sum += float(info.get("turnover", 0.0))
        dd = float(info.get("drawdown", 0.0))
        if np.isfinite(dd):
            max_drawdown = max(max_drawdown, dd)

        pnl_sum += float(info.get("pnl", 0.0))
        realized_pnl_sum += float(info.get("realized_pnl", 0.0))
        step_pnl_sum += float(info.get("step_pnl", 0.0))
        exposure_held_sum += float(info.get("exposure_held", 0.0))

        rparts = info.get("reward_parts", {}) or {}
        for k in reward_part_sums:
            reward_part_sums[k] += float(rparts.get(k, 0.0))
        steps += 1

    final_capital = float(raw.capital)
    return_pct = (final_capital - initial_capital) / max(initial_capital, 1e-9) * 100.0
    mean_reward_per_step = total_reward / max(steps, 1)

    pnl_abs = final_capital - initial_capital
    mean_exposure_held = exposure_held_sum / max(steps, 1)

    reward_part_means = {
        f"reward_{k}_mean": float(v / max(steps, 1))
        for k, v in reward_part_sums.items()
    }

    rm_metrics = (
        rm_trace_delta(rm_trace_state, rm_snapshot)
        if (rm_trace_state is not None and rm_snapshot is not None)
        else {
            "rm_steps": 0,
            "rm_accepts": 0,
            "rm_accept_rate": 0.0,
            "rm_mean_k_final": 0.0,
            "rm_k_final_min": 0.0,
            "rm_k_final_max": 0.0,
            "rm_veto_counts": {},
        }
    )

    return {
        "total_reward": total_reward,
        "episode_length": steps,
        "mean_reward_per_step": mean_reward_per_step,
        "return_pct": return_pct,
        "pnl_abs": pnl_abs,
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "sum_pnl_logged": pnl_sum,
        "sum_realized_pnl": realized_pnl_sum,
        "sum_step_pnl": step_pnl_sum,
        "max_drawdown": max_drawdown,
        "cumulative_turnover": turnover_sum,
        "mean_exposure_held": mean_exposure_held,
        "invalid_actions": invalid_actions,
        "ticker": raw.ticker,
        "rm_steps": rm_metrics["rm_steps"],
        "rm_accepts": rm_metrics["rm_accepts"],
        "rm_accept_rate": rm_metrics["rm_accept_rate"],
        "rm_mean_k_final": rm_metrics["rm_mean_k_final"],
        "rm_k_final_min": rm_metrics["rm_k_final_min"],
        "rm_k_final_max": rm_metrics["rm_k_final_max"],
        "rm_veto_counts": rm_metrics["rm_veto_counts"],
        **reward_part_means,
    }


def compute_eval_start_index(
    env: RiskEnv,
    ticker: str,
    base_start_index: int,
    max_steps_per_episode: int,
    episode_idx: int,
) -> int:
    base = int(base_start_index)
    try:
        ticker_len = int(len(env.df.loc[ticker]))
    except Exception:
        return base

    max_valid_start = max(base, ticker_len - int(max_steps_per_episode) - 2)
    if max_valid_start <= base:
        return base

    stride = max(1, int(max_steps_per_episode) // 2)
    candidate = base + int(episode_idx) * stride
    return int(min(candidate, max_valid_start))


def run_evaluation(
    env: RiskEnv,
    tickers: List[str],
    n_episodes: int,
    seed: int,
    rm: RiskManager,
    calibration_mode: str = "frozen",
    rm_trace_state: Optional[Dict[str, Any]] = None,
    start_index: int = 100,
    max_steps_per_episode: int = 500,
    eval_every_steps: int = 10_000,
    total_timesteps: Optional[int] = None,
) -> tuple[Dict[str, Any], RiskPolicy]:
    all_returns: List[float] = []
    all_rewards: List[float] = []
    all_lengths: List[int] = []
    all_drawdowns: List[float] = []
    all_rm_accept_rates: List[float] = []
    all_rm_kfinal: List[float] = []
    all_pnl_abs: List[float] = []
    all_turnover: List[float] = []
    all_exposure_held: List[float] = []

    all_reward_profit_mean: List[float] = []
    all_reward_dd_mean: List[float] = []
    all_reward_turn_mean: List[float] = []
    all_reward_smooth_mean: List[float] = []
    all_reward_bonus_mean: List[float] = []
    all_reward_no_exposure_mean: List[float] = []
    all_reward_risk_mean: List[float] = []
    all_reward_risk_excess_mean: List[float] = []
    rm_veto_counts_total: Dict[str, int] = {}
    episode_metrics: List[Dict[str, Any]] = []

    timesteps = total_timesteps if total_timesteps is not None else TRAIN_STEPS
    cal_mode = calibration_mode.lower().strip()
    if cal_mode == "frozen":
        rm.freeze_calibration()
    else:
        rm.set_training(True)

    policy = RiskPolicy(
        RiskPolicyConfig(
            algo="PPO",
            total_timesteps=timesteps,
            eval_every_steps=eval_every_steps,
            eval_episodes=3,
            verbose=1,
            seed=seed,
        )
    )
    policy.train(env, use_wandb=True)

    if getattr(policy, "env", None) is not None and hasattr(policy.env, "training"):
        policy.env.training = False
        if hasattr(policy.env, "norm_reward"):
            policy.env.norm_reward = False

    actual_train_steps = int(getattr(policy.model, "num_timesteps", timesteps))
    if actual_train_steps < timesteps:
        actual_train_steps = int(timesteps)

    if cal_mode == "train_only":
        rm.freeze_calibration()

    for ep in range(n_episodes):
        ticker_idx = ep % len(tickers)
        ticker_name = tickers[ticker_idx]
        episode_start_index = compute_eval_start_index(
            env=env,
            ticker=ticker_name,
            base_start_index=start_index,
            max_steps_per_episode=max_steps_per_episode,
            episode_idx=ep,
        )

        try:
            ep_metrics = run_episode(
                env=env,
                policy=policy,
                ticker_idx=ticker_idx,
                start_index=episode_start_index,
                max_steps=max_steps_per_episode,
                rm_trace_state=rm_trace_state,
            )
        except RuntimeError as e:
            episode_step = actual_train_steps + ep + 1
            wandb.log({
                "error": str(e),
                "aborted_ep": ep,
                "episode/seed": seed,
                "episode/start_index": episode_start_index,
            }, step=episode_step)
            continue

        all_returns.append(ep_metrics["return_pct"])
        all_rewards.append(ep_metrics["total_reward"])
        all_lengths.append(ep_metrics["episode_length"])
        all_drawdowns.append(ep_metrics["max_drawdown"])
        all_rm_accept_rates.append(ep_metrics["rm_accept_rate"])
        all_rm_kfinal.append(ep_metrics["rm_mean_k_final"])

        all_pnl_abs.append(ep_metrics["pnl_abs"])
        all_turnover.append(ep_metrics["cumulative_turnover"])
        all_exposure_held.append(ep_metrics["mean_exposure_held"])

        all_reward_profit_mean.append(ep_metrics["reward_profit_mean"])
        all_reward_dd_mean.append(ep_metrics["reward_dd_mean"])
        all_reward_turn_mean.append(ep_metrics["reward_turn_mean"])
        all_reward_smooth_mean.append(ep_metrics["reward_smooth_mean"])
        all_reward_bonus_mean.append(ep_metrics["reward_bonus_mean"])
        all_reward_no_exposure_mean.append(ep_metrics["reward_no_exposure_mean"])
        all_reward_risk_mean.append(ep_metrics["reward_risk_mean"])
        all_reward_risk_excess_mean.append(ep_metrics["reward_risk_excess_mean"])

        for reason, count in ep_metrics["rm_veto_counts"].items():
            rm_veto_counts_total[reason] = int(rm_veto_counts_total.get(reason, 0)) + int(count)

        episode_metrics.append(ep_metrics)

        episode_step = actual_train_steps + ep + 1
        ep_payload = {
            "episode/total_reward": ep_metrics["total_reward"],
            "episode/episode_length": ep_metrics["episode_length"],
            "episode/mean_reward_per_step": ep_metrics["mean_reward_per_step"],
            "episode/return_pct": ep_metrics["return_pct"],
            "episode/max_drawdown": ep_metrics["max_drawdown"],
            "episode/cumulative_turnover": ep_metrics["cumulative_turnover"],
            "episode/invalid_actions": ep_metrics["invalid_actions"],
            "episode/ticker": ep_metrics["ticker"],
            "episode/seed": seed,
            "episode/start_index": episode_start_index,
            "episode/rm_steps": ep_metrics["rm_steps"],
            "episode/rm_accepts": ep_metrics["rm_accepts"],
            "episode/rm_accept_rate": ep_metrics["rm_accept_rate"],
            "episode/rm_mean_k_final": ep_metrics["rm_mean_k_final"],
            "episode/rm_k_final_min": ep_metrics["rm_k_final_min"],
            "episode/rm_k_final_max": ep_metrics["rm_k_final_max"],
            "eval_epoch": ep + 1,
            "episode/pnl_abs": ep_metrics["pnl_abs"],
            "episode/initial_capital": ep_metrics["initial_capital"],
            "episode/final_capital": ep_metrics["final_capital"],
            "episode/sum_pnl_logged": ep_metrics["sum_pnl_logged"],
            "episode/sum_realized_pnl": ep_metrics["sum_realized_pnl"],
            "episode/sum_step_pnl": ep_metrics["sum_step_pnl"],
            "episode/mean_exposure_held": ep_metrics["mean_exposure_held"],

            "episode/reward_profit_mean": ep_metrics["reward_profit_mean"],
            "episode/reward_dd_mean": ep_metrics["reward_dd_mean"],
            "episode/reward_turn_mean": ep_metrics["reward_turn_mean"],
            "episode/reward_smooth_mean": ep_metrics["reward_smooth_mean"],
            "episode/reward_bonus_mean": ep_metrics["reward_bonus_mean"],
            "episode/reward_no_exposure_mean": ep_metrics["reward_no_exposure_mean"],
            "episode/reward_risk_mean": ep_metrics["reward_risk_mean"],
            "episode/reward_risk_excess_mean": ep_metrics["reward_risk_excess_mean"],
        }
        for reason, count in ep_metrics["rm_veto_counts"].items():
            ep_payload[f"episode/veto_count/{sanitize_metric_key(reason)}"] = int(count)

        wandb.log(ep_payload, step=episode_step)

    summary = {
        "n_episodes": n_episodes,
        "n_runs": len(all_returns),
        "seed": seed,
        "mean_return_pct": float(np.mean(all_returns)) if all_returns else 0.0,
        "std_return_pct": float(np.std(all_returns)) if all_returns else 0.0,
        "mean_total_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_total_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
        "mean_episode_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        "mean_max_drawdown": float(np.mean(all_drawdowns)) if all_drawdowns else 0.0,
        "rm_accept_rate": float(np.mean(all_rm_accept_rates)) if all_rm_accept_rates else 0.0,
        "rm_mean_k_final": float(np.mean(all_rm_kfinal)) if all_rm_kfinal else 0.0,
        "rm_veto_counts_total": rm_veto_counts_total,
        "episode_metrics": episode_metrics,
        "mean_pnl_abs": float(np.mean(all_pnl_abs)) if all_pnl_abs else 0.0,
        "std_pnl_abs": float(np.std(all_pnl_abs)) if all_pnl_abs else 0.0,
        "mean_cumulative_turnover": float(np.mean(all_turnover)) if all_turnover else 0.0,
        "mean_exposure_held": float(np.mean(all_exposure_held)) if all_exposure_held else 0.0,

        "mean_reward_profit": float(np.mean(all_reward_profit_mean)) if all_reward_profit_mean else 0.0,
        "mean_reward_dd": float(np.mean(all_reward_dd_mean)) if all_reward_dd_mean else 0.0,
        "mean_reward_turn": float(np.mean(all_reward_turn_mean)) if all_reward_turn_mean else 0.0,
        "mean_reward_smooth": float(np.mean(all_reward_smooth_mean)) if all_reward_smooth_mean else 0.0,
        "mean_reward_bonus": float(np.mean(all_reward_bonus_mean)) if all_reward_bonus_mean else 0.0,
        "mean_reward_no_exposure": float(np.mean(all_reward_no_exposure_mean)) if all_reward_no_exposure_mean else 0.0,
        "mean_reward_risk": float(np.mean(all_reward_risk_mean)) if all_reward_risk_mean else 0.0,
        "mean_reward_risk_excess": float(np.mean(all_reward_risk_excess_mean)) if all_reward_risk_excess_mean else 0.0,
    }

    summary_payload = {
        "summary/mean_return_pct": summary["mean_return_pct"],
        "summary/std_return_pct": summary["std_return_pct"],
        "summary/mean_total_reward": summary["mean_total_reward"],
        "summary/std_total_reward": summary["std_total_reward"],
        "summary/mean_episode_length": summary["mean_episode_length"],
        "summary/mean_max_drawdown": summary["mean_max_drawdown"],
        "summary/rm_accept_rate": summary["rm_accept_rate"],
        "summary/rm_mean_k_final": summary["rm_mean_k_final"],
        "summary/n_runs": summary["n_runs"],
        "summary/seed": seed,
        "summary/actual_train_steps": actual_train_steps,
        "summary/mean_pnl_abs": summary["mean_pnl_abs"],
        "summary/std_pnl_abs": summary["std_pnl_abs"],
        "summary/mean_cumulative_turnover": summary["mean_cumulative_turnover"],
        "summary/mean_exposure_held": summary["mean_exposure_held"],

        "summary/mean_reward_profit": summary["mean_reward_profit"],
        "summary/mean_reward_dd": summary["mean_reward_dd"],
        "summary/mean_reward_turn": summary["mean_reward_turn"],
        "summary/mean_reward_smooth": summary["mean_reward_smooth"],
        "summary/mean_reward_bonus": summary["mean_reward_bonus"],
        "summary/mean_reward_no_exposure": summary["mean_reward_no_exposure"],
        "summary/mean_reward_risk": summary["mean_reward_risk"],
        "summary/mean_reward_risk_excess": summary["mean_reward_risk_excess"],
    }
    for reason, count in rm_veto_counts_total.items():
        summary_payload[f"summary/veto_count/{sanitize_metric_key(reason)}"] = int(count)

    wandb.log(summary_payload, step=actual_train_steps + n_episodes + 1)

    return summary, policy


# ============================================================
# CLI helpers
# ============================================================

def parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def filter_assets(assets_arg: str) -> List[tuple]:
    if not assets_arg.strip():
        return ASSETS
    wanted = set(parse_csv_arg(assets_arg))
    out = [cfg for cfg in ASSETS if cfg[0] in wanted]
    if not out:
        raise ValueError(f"No assets matched --assets={assets_arg}. Available: {[a[0] for a in ASSETS]}")
    return out


def filter_modes(modes_arg: str) -> Dict[str, Dict[str, Any]]:
    if not modes_arg.strip():
        return ABLATIONS
    wanted = parse_csv_arg(modes_arg)
    bad = [m for m in wanted if m not in ABLATIONS]
    if bad:
        raise ValueError(f"Unknown mode(s): {bad}. Available: {sorted(ABLATIONS.keys())}")
    return {m: ABLATIONS[m] for m in wanted}


def filter_profiles(profiles_arg: str) -> List[Dict[str, Any]]:
    if not profiles_arg.strip():
        return SWEEP_SHORT_PROFILES
    wanted = parse_csv_arg(profiles_arg)
    bad = [p for p in wanted if p not in SWEEP_PROFILE_MAP]
    if bad:
        raise ValueError(f"Unknown profile(s): {bad}. Available: {sorted(SWEEP_PROFILE_MAP.keys())}")
    return [dict(SWEEP_PROFILE_MAP[p]) for p in wanted]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Reward-search version of risk-agent training/evaluation")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--start_index", type=int, default=100)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--eval_every_steps", type=int, default=10_000)
    parser.add_argument("--calibration_mode", type=str, default="frozen", choices=["frozen", "train_only", "dynamic"])

    parser.add_argument("--assets", type=str, default="", help="Comma-separated assets, e.g. BTC-USD")
    parser.add_argument("--modes", type=str, default="", help="Comma-separated modes, e.g. B3,B5")
    parser.add_argument("--profiles", type=str, default="", help="Comma-separated profiles, e.g. S2_tight,S6_risk_strict")

    parser.add_argument("--reward_presets", type=str, default="R0_base", help="Comma-separated reward preset names")
    parser.add_argument("--random_reward_search", action="store_true", default=False, help="Add random reward trials")
    parser.add_argument("--reward_trials", type=int, default=8, help="Number of random reward trials")
    parser.add_argument("--reward_seed", type=int, default=123, help="Seed for random reward search")

    parser.add_argument("--testrunning", action="store_true", default=False)

    args = parser.parse_args()
    ensure_wandb_login()

    train_steps = TRAIN_STEPS
    assets_to_run = filter_assets(args.assets)
    ablations_to_run = filter_modes(args.modes)
    profiles_to_run = filter_profiles(args.profiles)
    reward_candidates = build_reward_candidates(args)

    if args.testrunning:
        print("\n*** TEST RUNNING MODE: reduced iterations for quick validation ***\n")
        args.n_episodes = 1
        args.max_steps = 50
        args.eval_every_steps = 999_999
        train_steps = 200
        assets_to_run = assets_to_run[:1]
        ablations_to_run = {k: ablations_to_run[k] for k in list(ablations_to_run.keys())[:1]}
        profiles_to_run = profiles_to_run[:1]
        reward_candidates = reward_candidates[:2]

    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    for ticker, interval, period in assets_to_run:
        print("\n" + "=" * 80)
        print(f"LOADING DATA: {ticker} | {interval} | {period}")
        print("=" * 80)

        df = get_ohlcv_from_list([ticker], interval, period, "data/raw")
        if len(df) < 1000:
            print(f"Not enough data, skipping: {ticker} in {interval}_{period}")
            continue

        print("\tData is loaded and ready to be used.")

        df_t = df.loc[ticker]

        dir_count = 0
        vol_count = 0
        for t in range(150, min(len(df_t), 500)):
            if intent_provider_directional(t, df_t) is not None:
                dir_count += 1
            if intent_provider_volatility(t, df_t) is not None:
                vol_count += 1

        print("DEBUG directional candidates:", dir_count)
        print("DEBUG volatility candidates:", vol_count)

        for mode, cfg in ablations_to_run.items():
            for sweep_cfg in profiles_to_run:
                profile_name = str(sweep_cfg["name"])

                for reward_cfg in reward_candidates:
                    reward_name = str(reward_cfg["name"])

                    for seed_idx, seed in enumerate(seeds):
                        print("\n" + "-" * 70)
                        print(
                            f"TRAINING {ticker} | MODE {mode} | PROFILE {profile_name} "
                            f"| REWARD {reward_name} | SEED {seed} | CAL {args.calibration_mode}"
                        )
                        print("-" * 70)

                        wandb.init(
                            project=PROJECT_NAME,
                            name=f"{ticker}_{mode}_{profile_name}_{reward_name}_seed{seed}",
                            group=f"{ticker}_{interval}_{mode}_{profile_name}_reward_search",
                            mode=os.getenv("WANDB_MODE", "online"),
                            config={
                                "policy": f"{ticker}_{mode}",
                                "version": "v3_reward_search",
                                "n_episodes": args.n_episodes,
                                "n_seeds_total": len(seeds),
                                "seeds": seeds,
                                "seed": seed,
                                "seed_index": seed_idx,
                                "max_steps_per_episode": args.max_steps,
                                "start_index": args.start_index,
                                "eval_every_steps": args.eval_every_steps,
                                "train_steps": train_steps,
                                "environment": "RiskEnv",
                                "asset_ticker": ticker,
                                "asset_interval": interval,
                                "asset_period": period,
                                "ablation_mode": mode,
                                "use_quantiles": cfg["use_quantiles"],
                                "lambda_q": cfg["lambda_q"],
                                "lambda_reg": cfg["lambda_reg"],
                                "calibration_mode": args.calibration_mode,
                                "sweep_profile": profile_name,
                                "warn_kfinal": float(sweep_cfg["warn_kfinal"]),
                                "veto_kfinal": float(sweep_cfg["veto_kfinal"]),
                                "size_curve": float(sweep_cfg["size_curve"]),
                                "vol_requires_extreme": bool(sweep_cfg["vol_requires_extreme"]),
                                "reward_profile": reward_name,
                                "lam_pnl": float(reward_cfg["lam_pnl"]),
                                "lam_dd": float(reward_cfg["lam_dd"]),
                                "lam_turn": float(reward_cfg["lam_turn"]),
                                "lam_smooth": float(reward_cfg["lam_smooth"]),
                                "lam_bonus": float(reward_cfg["lam_bonus"]),
                                "lam_no_exposure": float(reward_cfg["lam_no_exposure"]),
                                "lam_risk": float(reward_cfg["lam_risk"]),
                            },
                            tags=[
                                "PPO",
                                "reward_search",
                                mode,
                                ticker,
                                f"profile_{profile_name}",
                                f"reward_{reward_name}",
                                f"calib_{args.calibration_mode}",
                                f"seed_{seed}",
                            ],
                        )
                        print_wandb_run_info()

                        rm = RiskManager(
                            RiskManagerConfig(
                                ablation_mode=mode,
                                lambda_reg=cfg["lambda_reg"],
                                lambda_q=cfg["lambda_q"],
                                warn_kfinal=float(sweep_cfg["warn_kfinal"]),
                                veto_kfinal=float(sweep_cfg["veto_kfinal"]),
                                size_curve=float(sweep_cfg["size_curve"]),
                                auto_calibrate=(args.calibration_mode != "frozen"),
                                rolling_calibration=True,
                                calibration_warmup=300,
                                rolling_window=800,
                                recalibrate_every=50,
                                calibration_smoothing=0.2,
                                calibration_quantile_warn=0.93,
                                calibration_quantile_veto=0.995,
                                calibration_min_gap=0.10,
                                calibration_max_veto=3.0,
                                vol_min_confidence=0.25,
                                vol_requires_extreme=bool(sweep_cfg["vol_requires_extreme"]),
                                vol_ultra_extra=0.20,
                                vol_size_mult=0.70,
                                vol_sl_mult=1.20,
                                vol_tp_mult=1.25,
                                vol_ultra_size_mult=0.35,
                                vol_ultra_sl_mult=1.35,
                                vol_ultra_tp_mult=1.40,
                            )
                        )

                        if args.calibration_mode == "frozen":
                            rm.freeze_calibration()

                        qra_cache: Dict[int, Any] = {}
                        rm_trace_state = new_rm_trace_state()

                        decision_p = lambda t, ticker_name, df_full: decision_provider(
                            t,
                            df_full.loc[ticker_name],
                            cfg["use_quantiles"],
                            rm,
                            qra_cache=qra_cache,
                            rm_trace_state=rm_trace_state,
                        )

                        env = RiskEnv(
                            df_ohlcv=df,
                            tickers=[ticker],
                            cfg=RiskEnvConfig(
                                initial_capital=10_000.0,
                                max_position_units=1.0,
                                base_sl_atr=1.5,
                                base_tp_atr=2.0,
                                min_size_mult=0.3,
                                rew_cfg=RewardConfig(
                                    lam_pnl=float(reward_cfg["lam_pnl"]),
                                    lam_dd=float(reward_cfg["lam_dd"]),
                                    lam_turn=float(reward_cfg["lam_turn"]),
                                    lam_smooth=float(reward_cfg["lam_smooth"]),
                                    lam_bonus=float(reward_cfg["lam_bonus"]),
                                    lam_no_exposure=float(reward_cfg["lam_no_exposure"]),
                                    lam_risk=float(reward_cfg["lam_risk"]),
                                ),
                            ),
                            decision_provider=decision_p,
                        )

                        wandb.config.update({
                            "obs_dim": env.obs_dim,
                            "n_tickers": 1,
                            "tickers": [ticker],
                        })

                        summary, policy = run_evaluation(
                            env=env,
                            tickers=[ticker],
                            n_episodes=args.n_episodes,
                            seed=seed,
                            rm=rm,
                            calibration_mode=args.calibration_mode,
                            rm_trace_state=rm_trace_state,
                            start_index=args.start_index,
                            max_steps_per_episode=args.max_steps,
                            eval_every_steps=args.eval_every_steps,
                            total_timesteps=train_steps,
                        )

                        model_name = f"ppo_{ticker}_{interval}_{mode}_{profile_name}_{reward_name}_seed{seed}"
                        model_path = os.path.join(BASE_DIR, model_name)
                        policy.model.save(model_path)
                        print(f"Model saved to: {model_path}")

                        wandb.finish()


if __name__ == "__main__":
    main()