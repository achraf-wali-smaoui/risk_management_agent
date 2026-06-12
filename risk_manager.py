# ============================================================
# Risk Manager — Ablations B1..B5
# Strategy-aware veto (directional vs volatility)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class TradeIntent:
    direction: str                    # "long" | "short" | "flat"
    baseline_size: float
    confidence: float
    strategy_type: str = "directional"   # "directional" | "volatility"
    meta: Optional[Dict[str, Any]] = None


@dataclass
class RiskAction:
    scale_size: float
    scale_sl: float
    scale_tp: float
    accept_reject: int


@dataclass
class RiskManagerOutput:
    action: RiskAction
    k_final: float
    diagnostics: Dict[str, Any]
    explanation: Dict[str, Any]


# ----------------------------
# Config
# ----------------------------

@dataclass
class RiskManagerConfig:
    # --- fusion weights (convex) ---
    lambda_reg: float = 0.5
    lambda_q: float = 0.5

    # --- gating ---
    min_confidence: float = 0.05

    # --- volatility strategy gating ---
    vol_min_confidence: float = 0.20
    vol_requires_extreme: bool = True
    vol_ultra_extra: float = 0.15



    # If auto_calibrate=True, these will be overwritten dynamically
    warn_kfinal: float = 1.15
    veto_kfinal: float = 1.35

    auto_calibrate: bool = True
    calibration_warmup: int = 200
    calibration_quantile_warn: float = 0.93
    calibration_quantile_veto: float = 0.995
    calibration_min_gap: float = 0.1
    calibration_max_veto: float = 3.0

    rolling_calibration: bool = True
    rolling_window: int = 400
    recalibrate_every: int = 50
    calibration_smoothing: float = 0.2




    # --- scaling curves ---
    size_curve: float = 2.0
    sl_curve: float = 1.0
    tp_curve: float = 1.0

    # --- bounds ---
    size_min: float = 0.0
    size_max: float = 1.0
    sl_min: float = 0.8
    sl_max: float = 1.6
    tp_min: float = 0.7
    tp_max: float = 1.6

    tp_reduce_in_high_risk: bool = True

    # --- volatility strategy scaling ---
    vol_size_mult: float = 0.75
    vol_sl_mult: float = 1.15
    vol_tp_mult: float = 1.25

    # ultra extreme scaling (micro exposure)
    vol_ultra_size_mult: float = 0.40
    vol_ultra_sl_mult: float = 1.30
    vol_ultra_tp_mult: float = 1.35


    # --- ablation mode ---
    ablation_mode: str = "B5"   # B1..B5

    # --- fuzzy mapping ---
    fuzzy_k_low: float = 1.00
    fuzzy_k_mid: float = 1.30
    fuzzy_k_high: float = 1.75


# ----------------------------
# Helpers
# ----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _risk_to_size_scale(k: float, warn: float, curve: float) -> float:
    if k <= warn:
        return 1.0
    return 1.0 / (1.0 + (k - warn) * curve)


def _risk_to_sl_scale(k: float, lo: float, hi: float, curve: float) -> float:
    x = 1.0 + (k - 1.0) * curve
    return _clamp(x, lo, hi)


def _risk_to_tp_scale(k: float, lo: float, hi: float, curve: float, reduce: bool) -> float:
    if not reduce:
        x = 1.0 + (k - 1.0) * curve
        return _clamp(x, lo, hi)
    x = 1.0 / (1.0 + (k - 1.0) * curve)
    return _clamp(x, lo, hi)


def _tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)


# ----------------------------
# RiskManager
# ----------------------------

class RiskManager:
    def __init__(self, cfg: Optional[RiskManagerConfig] = None):
        self.cfg = cfg or RiskManagerConfig()

        self._k_history = []
        self._step_counter = 0
        self._calibration_enabled = True   # training by default




        s = max(self.cfg.lambda_reg + self.cfg.lambda_q, 1e-9)
        self._lambda_reg = self.cfg.lambda_reg / s
        self._lambda_q = self.cfg.lambda_q / s

    # ---------- Fusion ----------
    def fuse_risk_convex(
        self, regime: float, q: float, use_regime: bool, use_quantile: bool
    ) -> float:
        w_reg = self._lambda_reg if use_regime else 0.0
        w_q = self._lambda_q if use_quantile else 0.0
        s = max(w_reg + w_q, 1e-9)
        return (w_reg / s) * regime + (w_q / s) * q

    def fuse_risk_fuzzy(self, regime: float, q: float) -> Tuple[float, Dict[str, Any]]:
        vol_low = _tri(regime, 0.9, 1.0, 1.2)
        vol_med = _tri(regime, 1.1, 1.3, 1.6)
        vol_high = _tri(regime, 1.4, 1.7, 2.2)

        q_low = _tri(q, 0.9, 1.0, 1.2)
        q_med = _tri(q, 1.1, 1.3, 1.6)
        q_high = _tri(q, 1.4, 1.7, 2.2)

        r_low = min(vol_low, q_low)
        r_med = min(vol_med, q_med)
        r_high = max(vol_high, q_high)

        num = r_low * 0.15 + r_med * 0.55 + r_high * 0.90
        den = r_low + r_med + r_high + 1e-12
        score = num / den

        if score <= 0.55:
            alpha = score / 0.55
            k = (1 - alpha) * self.cfg.fuzzy_k_low + alpha * self.cfg.fuzzy_k_mid
        else:
            alpha = (score - 0.55) / 0.45
            k = (1 - alpha) * self.cfg.fuzzy_k_mid + alpha * self.cfg.fuzzy_k_high

        return float(k), {"risk_score": float(score)}
    
    def set_training(self, is_training: bool) -> None:
        """Enable rolling calibration during training; freeze during evaluation."""
        self._calibration_enabled = bool(is_training)

    def freeze_calibration(self) -> None:
        """Convenience: freeze thresholds (no updates)."""
        self._calibration_enabled = False

    # ---------- Decision ----------
    def decide(self, risk_out: Any, intent: TradeIntent) -> RiskManagerOutput:
        regime_raw = _safe_float(getattr(risk_out, "regime_factor", 1.0), 1.0)
        q_raw = _safe_float(getattr(risk_out, "q_factor", 1.0), 1.0)

        # Convert "risk reduction" factors (smaller = riskier)
        # into "risk intensity" (bigger = riskier)
        regime = 1.0 / max(regime_raw, 1e-9)
        q = 1.0 / max(q_raw, 1e-9)


        veto = []
        if intent.direction == "flat":
            veto.append("flat_direction")

        mode = self.cfg.ablation_mode.upper()
        fuzzy_trace = {}

        # ================= ABLATIONS =================
        if mode == "B1":
            # static baseline (no risk)
            k_final = 1.0
            use_regime = use_quantile = False
            fuzzy = False

        elif mode == "B2":
            k_final = self.fuse_risk_convex(regime, q, True, False)
            use_regime, use_quantile = True, False
            fuzzy = False

        elif mode == "B3":
            k_final = self.fuse_risk_convex(regime, q, False, True)
            use_regime, use_quantile = False, True
            fuzzy = False

        elif mode == "B4":
            k_final = self.fuse_risk_convex(regime, q, True, True)
            use_regime, use_quantile = True, True
            fuzzy = False

        elif mode == "B5":
            k_final, fuzzy_trace = self.fuse_risk_fuzzy(regime, q)
            use_regime, use_quantile = True, True
            fuzzy = True

        else:
            raise ValueError(f"Unknown ablation_mode: {mode}")
        


        # ---------- Rolling Automatic Threshold Calibration (train-only) ----------
        if self.cfg.auto_calibrate and self._calibration_enabled:

            self._k_history.append(float(k_final))
            self._step_counter += 1

            # Rolling window
            if self.cfg.rolling_calibration and (len(self._k_history) > self.cfg.rolling_window):
                self._k_history.pop(0)

            # Periodic recalibration (after warmup)
            if (
                len(self._k_history) >= self.cfg.calibration_warmup
                and self._step_counter % self.cfg.recalibrate_every == 0
            ):
                import numpy as np

                arr = np.array(self._k_history)

                warn_new = float(np.percentile(arr, self.cfg.calibration_quantile_warn * 100))
                veto_new = float(np.percentile(arr, self.cfg.calibration_quantile_veto * 100))

                # Enforce separation + clamp
                veto_new = max(veto_new, warn_new + self.cfg.calibration_min_gap)
                veto_new = min(veto_new, self.cfg.calibration_max_veto)

                # Smooth thresholds (EMA)
                a = float(self.cfg.calibration_smoothing)
                self.cfg.warn_kfinal = (1 - a) * self.cfg.warn_kfinal + a * warn_new
                self.cfg.veto_kfinal = (1 - a) * self.cfg.veto_kfinal + a * veto_new




        # ---------- Strategy-aware gating ----------
        strategy = intent.strategy_type
        warn = self.cfg.warn_kfinal
        veto_th = self.cfg.veto_kfinal
        min_conf = self.cfg.min_confidence

        ultra = False

        # Directional logic
        if strategy == "directional":
            if k_final >= veto_th * 1.1:
                veto.append("extreme_risk_directional")

        # Volatility logic
        elif strategy == "volatility":

            # require extreme regime if enabled
            if self.cfg.vol_requires_extreme and k_final < veto_th:
                veto.append("vol_not_extreme_enough")

            # ultra extreme zone
            if k_final >= veto_th + self.cfg.vol_ultra_extra:
                ultra = True

            # stricter confidence
            if intent.confidence < max(min_conf, self.cfg.vol_min_confidence):
                veto.append("vol_low_confidence")


        accept = 0 if veto else 1

        raw_size = _risk_to_size_scale(k_final, self.cfg.warn_kfinal, self.cfg.size_curve)
        scale_size = _clamp(raw_size, self.cfg.size_min, self.cfg.size_max)
        if not accept:
            scale_size = 0.0

        scale_sl = _clamp(
            _risk_to_sl_scale(k_final, self.cfg.sl_min, self.cfg.sl_max, self.cfg.sl_curve),
            self.cfg.sl_min,
            self.cfg.sl_max,
        )

        scale_tp = _clamp(
            _risk_to_tp_scale(
                k_final,
                self.cfg.tp_min,
                self.cfg.tp_max,
                self.cfg.tp_curve,
                self.cfg.tp_reduce_in_high_risk,
            ),
            self.cfg.tp_min,
            self.cfg.tp_max,
        )

        # ---------- Strategy-dependent scaling ----------
        if strategy == "volatility" and accept:

            scale_size *= self.cfg.vol_size_mult
            scale_sl   *= self.cfg.vol_sl_mult
            scale_tp   *= self.cfg.vol_tp_mult

            # ultra extreme -> micro exposure
            if ultra:
                scale_size *= self.cfg.vol_ultra_size_mult
                scale_sl   *= self.cfg.vol_ultra_sl_mult
                scale_tp   *= self.cfg.vol_ultra_tp_mult

            # re-clamp
            scale_size = _clamp(scale_size, self.cfg.size_min, self.cfg.size_max)
            scale_sl   = _clamp(scale_sl, self.cfg.sl_min, self.cfg.sl_max)
            scale_tp   = _clamp(scale_tp, self.cfg.tp_min, self.cfg.tp_max)


        return RiskManagerOutput(
            action=RiskAction(scale_size, scale_sl, scale_tp, accept),
            k_final=float(k_final),
            diagnostics={
                "mode": mode,
                "k_final": float(k_final),
                "use_regime": use_regime,
                "use_quantile": use_quantile,
                "fuzzy": fuzzy,
            },
            explanation={
                "decision": "ACCEPT" if accept else "REJECT",
                "strategy_type": intent.strategy_type,
                "veto_reasons": veto,
            },
        )