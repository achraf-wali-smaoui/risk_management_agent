"""
Fuzzy Hybrid Risk Aggregator for XDTA.

Goal
----
Aggregate heterogeneous uncertainty descriptors into:
- risk_score in [0,1]
- risk_label in {"low","moderate","high","very_high"}

This module is intentionally lightweight:
- simple triangular membership functions
- compact expert-inspired rule base
- weighted-average defuzzification (stable, fast)

Inputs (normalized or near-normalized)
--------------------------------------
vol_level : float in [0,1]   (current volatility / regime intensity)
tail_risk : float in [0,1]   (forward-looking tail / quantile risk intensity)
risk_delta: float in [-1,1]  (short-horizon change in tail risk; + means increasing)

You can adapt the normalization strategy in RiskManager depending on what
your QuantileRiskAgent provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal

RiskLabel = Literal["low", "moderate", "high", "very_high"]


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def tri_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function:
      a --- b --- c
    mu=0 outside [a,c], mu=1 at b
    """
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / max(b - a, 1e-12)
    return (c - x) / max(c - b, 1e-12)


@dataclass
class FuzzyRiskConfig:
    # --- Membership breakpoints for vol_level and tail_risk in [0,1] ---
    # Low  : peak at ~0.2
    # Med  : peak at ~0.5
    # High : peak at ~0.8
    vol_low: Tuple[float, float, float] = (0.0, 0.2, 0.4)
    vol_med: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    vol_high: Tuple[float, float, float] = (0.6, 0.8, 1.0)

    tail_low: Tuple[float, float, float] = (0.0, 0.2, 0.4)
    tail_med: Tuple[float, float, float] = (0.2, 0.5, 0.8)
    tail_high: Tuple[float, float, float] = (0.6, 0.8, 1.0)

    # risk_delta in [-1,1]
    # dec : peak -0.6, flat : 0, inc : +0.6
    d_dec: Tuple[float, float, float] = (-1.0, -0.6, -0.2)
    d_flat: Tuple[float, float, float] = (-0.3, 0.0, 0.3)
    d_inc: Tuple[float, float, float] = (0.2, 0.6, 1.0)

    # Output crisp values (Sugeno-style) for stability
    out_low: float = 0.20
    out_mod: float = 0.50
    out_high: float = 0.75
    out_vhigh: float = 0.90

    # Label thresholds on risk_score
    th_low: float = 0.33
    th_high: float = 0.66
    th_vhigh: float = 0.85


class FuzzyRiskAggregator:
    def __init__(self, cfg: FuzzyRiskConfig | None = None) -> None:
        self.cfg = cfg or FuzzyRiskConfig()

    def _memberships(self, vol_level: float, tail_risk: float, risk_delta: float) -> Dict[str, float]:
        v = _clamp(vol_level, 0.0, 1.0)
        t = _clamp(tail_risk, 0.0, 1.0)
        d = _clamp(risk_delta, -1.0, 1.0)

        mu = {
            "v_low": tri_mf(v, *self.cfg.vol_low),
            "v_med": tri_mf(v, *self.cfg.vol_med),
            "v_high": tri_mf(v, *self.cfg.vol_high),

            "t_low": tri_mf(t, *self.cfg.tail_low),
            "t_med": tri_mf(t, *self.cfg.tail_med),
            "t_high": tri_mf(t, *self.cfg.tail_high),

            "d_dec": tri_mf(d, *self.cfg.d_dec),
            "d_flat": tri_mf(d, *self.cfg.d_flat),
            "d_inc": tri_mf(d, *self.cfg.d_inc),
        }
        return mu

    def infer(self, vol_level: float, tail_risk: float, risk_delta: float = 0.0) -> Dict[str, object]:
        """
        Return:
          {
            "risk_score": float in [0,1],
            "risk_label": RiskLabel,
            "memberships": {...},
            "rule_activations": {...}
          }
        """
        mu = self._memberships(vol_level, tail_risk, risk_delta)

        # ---- Compact rule base (Sugeno outputs) ----
        # AND is min; rule firing strength = min(memberships)
        rules = {}

        # R1: vol high & tail high & increasing -> very high
        rules["R1_vhigh"] = min(mu["v_high"], mu["t_high"], mu["d_inc"])
        # R2: vol high & tail high (even if flat) -> high
        rules["R2_high"] = min(mu["v_high"], mu["t_high"], max(mu["d_flat"], mu["d_dec"]))
        # R3: vol high & tail med & increasing -> high
        rules["R3_high"] = min(mu["v_high"], mu["t_med"], mu["d_inc"])
        # R4: vol med & tail high -> high
        rules["R4_high"] = min(mu["v_med"], mu["t_high"])
        # R5: vol med & tail med & increasing -> moderate/high (we set high for safety)
        rules["R5_high"] = min(mu["v_med"], mu["t_med"], mu["d_inc"])
        # R6: vol med & tail med & flat -> moderate
        rules["R6_mod"] = min(mu["v_med"], mu["t_med"], mu["d_flat"])
        # R7: vol low & tail med -> moderate
        rules["R7_mod"] = min(mu["v_low"], mu["t_med"])
        # R8: vol low & tail low -> low
        rules["R8_low"] = min(mu["v_low"], mu["t_low"])
        # R9: tail low & decreasing -> low
        rules["R9_low"] = min(mu["t_low"], mu["d_dec"])
        # R10: vol med & tail low -> low/moderate (use moderate to avoid underestimating)
        rules["R10_mod"] = min(mu["v_med"], mu["t_low"])

        # Defuzzification: weighted average of crisp outputs
        num = (
            rules["R1_vhigh"] * self.cfg.out_vhigh
            + rules["R2_high"] * self.cfg.out_high
            + rules["R3_high"] * self.cfg.out_high
            + rules["R4_high"] * self.cfg.out_high
            + rules["R5_high"] * self.cfg.out_high
            + rules["R6_mod"] * self.cfg.out_mod
            + rules["R7_mod"] * self.cfg.out_mod
            + rules["R8_low"] * self.cfg.out_low
            + rules["R9_low"] * self.cfg.out_low
            + rules["R10_mod"] * self.cfg.out_mod
        )
        den = sum(rules.values()) + 1e-12
        risk_score = _clamp(num / den, 0.0, 1.0)

        # Label
        if risk_score >= self.cfg.th_vhigh:
            label: RiskLabel = "very_high"
        elif risk_score >= self.cfg.th_high:
            label = "high"
        elif risk_score >= self.cfg.th_low:
            label = "moderate"
        else:
            label = "low"

        return {
            "risk_score": float(risk_score),
            "risk_label": label,
            "memberships": mu,
            "rule_activations": rules,
        }
