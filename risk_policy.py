# ============================================================
# risk_policy.py
# Reinforcement Learning policy for RiskEnv (Risk management only)
#
# Strategy-aware update (STEP 3):
# - RL action = 3D: [size_mult, sl_mult, tp_mult]
# - accept/reject is NOT learned; it comes from RiskManager via intent_provider.
# - We optionally APPEND a strategy flag to the observation:
#     0.0 = directional, 1.0 = volatility
#   so the RL controller can learn distinct behaviors per strategy.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict
import copy
import numpy as np

# Gymnasium is preferred for SB3>=2.x
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Missing dependency: install Gymnasium with `uv add gymnasium`."
    ) from exc

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    PPO = None
    SAC = None
    DummyVecEnv = None
    VecNormalize = None
    BaseCallback = object


# ============================================================
# Config
# ============================================================

@dataclass
class RiskPolicyConfig:
    # Algorithm
    algo: str = "PPO"
    seed: int = 42
    total_timesteps: int = 50_000
    net_arch: Tuple[int, int] = (64, 64)

    # PPO
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.98
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 256
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 1e-3

    # SAC
    sac_learning_rate: float = 3e-4
    sac_gamma: float = 0.98
    sac_buffer_size: int = 200_000
    sac_batch_size: int = 256
    sac_tau: float = 0.005

    # Eval # 321
    verbose: int = 1
    eval_every_steps: int = 10_000  # smaller = more eval checkpoints in wandb (e.g. 12 points for 120k steps)
    eval_episodes: int = 3
    eval_warmup_stride: int = 200  # shift eval episodes to different temporal regions

    # Action bounds for the RL risk controller (multipliers)
    size_mult_min: float = 0.0
    size_mult_max: float = 1.0
    sl_mult_min: float = 0.5
    sl_mult_max: float = 2.0
    tp_mult_min: float = 0.5
    tp_mult_max: float = 2.0

    # -------- Strategy-aware observation augmentation --------
    # If True, append one scalar to obs: 0=directional, 1=volatility
    append_strategy_flag: bool = True
    strategy_flag_default: float = 0.0  # fallback if env doesn't expose intent


# ============================================================
# Optional: action conversion (for logging)
# ============================================================

def vector_to_risk_action(a: np.ndarray, cfg: RiskPolicyConfig) -> dict:
    """Convert 3D RL vector to a readable dict (NO accept/reject here)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size != 3:
        raise ValueError(f"Action must be 3D, got {a.size}")

    return {
        "size_mult": float(np.clip(a[0], cfg.size_mult_min, cfg.size_mult_max)),
        "sl_mult":   float(np.clip(a[1], cfg.sl_mult_min, cfg.sl_mult_max)),
        "tp_mult":   float(np.clip(a[2], cfg.tp_mult_min, cfg.tp_mult_max)),
    }


# ============================================================
# Gym wrapper
# ============================================================

class RiskEnvGymWrapper(gym.Env):
    """Wraps the custom RiskEnv to a Gymnasium API so it can be used with SB3.

    Strategy-aware:
    - Optionally appends a strategy flag to the observation.
    - Works via duck-typing: tries to read last intent from env.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: Any, cfg: RiskPolicyConfig):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for RiskEnvGymWrapper. Please install gymnasium.")
        super().__init__()
        self.env = env
        self.cfg = cfg

        # ---- Action space (float32): [size_mult, sl_mult, tp_mult] ----
        self.action_space = spaces.Box(
            low=np.array([cfg.size_mult_min, cfg.sl_mult_min, cfg.tp_mult_min], dtype=np.float32),
            high=np.array([cfg.size_mult_max, cfg.sl_mult_max, cfg.tp_mult_max], dtype=np.float32),
            dtype=np.float32,
        )

        # ---- Observation space ----
        obs = self._reset_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    # -------- Strategy flag extraction (duck-typed) --------
    def _extract_strategy_type(self) -> Optional[str]:
        """
        Attempts to read the most recent intent/strategy_type from the underlying env.
        This is intentionally robust to different implementations.
        """
        e = self.env

        # 1) method
        if hasattr(e, "get_last_intent") and callable(getattr(e, "get_last_intent")):
            try:
                intent = e.get_last_intent()
                if isinstance(intent, dict):
                    return intent.get("strategy_type")
                return getattr(intent, "strategy_type", None)
            except Exception:
                pass

        # 2) attribute: last_intent / intent
        for attr in ("last_intent", "intent"):
            if hasattr(e, attr):
                try:
                    intent = getattr(e, attr)
                    if isinstance(intent, dict):
                        return intent.get("strategy_type")
                    return getattr(intent, "strategy_type", None)
                except Exception:
                    pass

        return None

    def _strategy_flag(self) -> float:
        if not self.cfg.append_strategy_flag:
            return 0.0

        st = self._extract_strategy_type()
        if isinstance(st, str):
            st = st.lower().strip()
            if st == "volatility":
                return 1.0
            if st == "directional":
                return 0.0

        return float(self.cfg.strategy_flag_default)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not self.cfg.append_strategy_flag:
            return obs
        flag = np.array([self._strategy_flag()], dtype=np.float32)
        return np.concatenate([obs, flag], axis=0)

    def _reset_obs(self, options: Optional[dict] = None) -> np.ndarray:
        if options is not None and ("start_index" in options or "idx_ticker" in options):
            start_index = int(options.get("start_index", 0))
            idx_ticker = int(options.get("idx_ticker", 0))
            out = self.env.reset(start_index=start_index, idx_ticker=idx_ticker)
        else:
            out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        return self._augment_obs(obs)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = self._reset_obs(options=options)
        return obs, {}

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size != 3:
            raise ValueError(f"Action must have 3 dims, got {a.size}")

        # clip to bounds
        a[0] = np.clip(a[0], self.cfg.size_mult_min, self.cfg.size_mult_max)
        a[1] = np.clip(a[1], self.cfg.sl_mult_min, self.cfg.sl_mult_max)
        a[2] = np.clip(a[2], self.cfg.tp_mult_min, self.cfg.tp_mult_max)

        obs, reward, done, info = self.env.step(a)
        obs = self._augment_obs(obs)

        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


# ============================================================
# Eval callback (API-robust)
# ============================================================

class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_every_steps=10_000,
        n_episodes=3,
        verbose=1,
        use_wandb: bool = False,
        eval_seed_base: Optional[int] = None,
        eval_warmup_stride: int = 200,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every_steps = int(eval_every_steps)
        self.n_episodes = int(n_episodes)
        self.use_wandb = bool(use_wandb)
        self.eval_warmup_stride = max(0, int(eval_warmup_stride))
        self._last_eval_step = -1

    def _reset_eval_env(self, options: Optional[dict] = None):
        if options:
            set_options = getattr(self.eval_env, "set_options", None)
            if callable(set_options):
                set_options(options)
                return self.eval_env.reset()
        return self.eval_env.reset()

    def _on_step(self) -> bool:
        current_step = int(getattr(self, "num_timesteps", self.n_calls))
        should_eval = (
            self.eval_every_steps > 0
            and current_step > 0
            and (current_step % self.eval_every_steps == 0)
            and (current_step != self._last_eval_step)
        )
        if should_eval:
            self._last_eval_step = current_step
            rewards = []
            for ep_idx in range(self.n_episodes):
                warmup_steps = ep_idx * self.eval_warmup_stride
                reset_out = self._reset_eval_env({"start_index": warmup_steps})
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

                # Use deterministic warmup offsets so each episode measures
                # a different part of the trajectory (non-trivial std).
               

               
               # advanced = 0
               # while advanced < warmup_steps:
                #    a, _ = self.model.predict(obs, deterministic=True)
                #    step_out = self.eval_env.step(a)
                 #   if len(step_out) == 5:
                 #       obs, _r, terminated, truncated, _info = step_out
                 #       done_warmup = bool(terminated) or bool(truncated)
                  #  else:
                  #      obs, _r, done_warmup, _info = step_out
                  #      done_warmup = bool(done_warmup)
                  #  advanced += 1
                  #  if done_warmup:
                  #      reset_out = self.eval_env.reset()
                  #      obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

                done = False
                total = 0.0
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    step_out = self.eval_env.step(a)

                    if len(step_out) == 5:
                        obs, r, terminated, truncated, _info = step_out
                        done = bool(terminated) or bool(truncated)
                    else:
                        obs, r, done, _info = step_out
                        done = bool(done)

                    total += float(r)

                rewards.append(total)

            mean_reward = float(np.mean(rewards))
            if self.verbose:
                print(f"[Eval] mean={mean_reward:.4f}")

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": float(np.std(rewards)),
                        "eval/n_episodes": self.n_episodes,
                    }, step=current_step)
                except Exception:
                    pass

        return True


# ============================================================
# MAIN POLICY
# ============================================================

class RiskPolicy:
    def __init__(self, cfg: RiskPolicyConfig):
        self.cfg = cfg
        self.algo = cfg.algo.upper()
        self.model = None
        self.env = None

    def build_model(self, env):
        if PPO is None or SAC is None:
            raise ImportError("stable-baselines3 is required to build the model.")

        policy_kwargs = dict(net_arch=list(self.cfg.net_arch))

        if self.algo == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.cfg.ppo_learning_rate,
                gamma=self.cfg.ppo_gamma,
                n_steps=self.cfg.ppo_n_steps,
                batch_size=self.cfg.ppo_batch_size,
                clip_range=self.cfg.ppo_clip_range,
                ent_coef=self.cfg.ppo_ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=self.cfg.verbose,
                seed=self.cfg.seed,
            )
        elif self.algo == "SAC":
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=self.cfg.sac_learning_rate,
                gamma=self.cfg.sac_gamma,
                buffer_size=self.cfg.sac_buffer_size,
                batch_size=self.cfg.sac_batch_size,
                tau=self.cfg.sac_tau,
                policy_kwargs=policy_kwargs,
                verbose=self.cfg.verbose,
                seed=self.cfg.seed,
            )
        else:
            raise ValueError(f"Unsupported algo: {self.algo}")

    def train(self, raw_env: Any, use_wandb: bool = False):
        """Train PPO / SAC on RiskEnv using Gym wrapper + VecNormalize.

        use_wandb: if True, PeriodicEvalCallback will log eval/mean_reward, eval/std_reward to wandb.
        """
        if DummyVecEnv is None or VecNormalize is None:
            raise ImportError("stable-baselines3 (with vec_env) is required for training.")

        # Keep train/eval environment states fully separated.
        # Using the same raw env instance for both can leak resets/steps from eval callback into training.
        try:
            train_raw_env = copy.deepcopy(raw_env)
            eval_raw_env = copy.deepcopy(raw_env)
        except Exception as e:
            if self.cfg.verbose:
                print(f"[RiskPolicy][WARN] deepcopy(raw_env) failed ({e}); train/eval will share state.")
            train_raw_env = raw_env
            eval_raw_env = raw_env

        # 1) Training env
        train_env = DummyVecEnv([lambda: RiskEnvGymWrapper(train_raw_env, self.cfg)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        # 2) Model
        self.build_model(train_env)

        # 3) Eval env (share obs normalization stats)
        eval_env = DummyVecEnv([lambda: RiskEnvGymWrapper(eval_raw_env, self.cfg)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.training = False

        # 4) Callback (optional wandb logging during eval)
        cb = PeriodicEvalCallback(
            eval_env,
            self.cfg.eval_every_steps,
            self.cfg.eval_episodes,
            self.cfg.verbose,
            use_wandb=use_wandb,
            eval_warmup_stride=self.cfg.eval_warmup_stride,
        )

        # 5) Learn
        self.model.learn(total_timesteps=self.cfg.total_timesteps, callback=cb)

        self.env = train_env
        return self.model


# ============================================================
# Fallback random rollout
# ============================================================

def random_rollout(raw_env, cfg: RiskPolicyConfig, n_steps=50, seed=0):
    rng = np.random.default_rng(seed)
    env = RiskEnvGymWrapper(raw_env, cfg)
    obs, _ = env.reset()
    total = 0.0
    for i in range(n_steps):
        a = rng.uniform(env.action_space.low, env.action_space.high)
        step_out = env.step(a)
        if len(step_out) == 5:
            obs, r, terminated, truncated, _info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, r, done, _info = step_out
            done = bool(done)
        total += float(r)
        print(f"[random] step={i:03d} r={float(r):+.4f}")
        if done:
            obs, _ = env.reset()
    return total
