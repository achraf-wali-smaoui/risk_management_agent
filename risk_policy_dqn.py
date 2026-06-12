# ============================================================
# risk_policy_dqn.py 
# Deep Q-Network (DQN) policy for RiskEnv (Risk management only)
#
# OPTION B DESIGN (agreed):
# - RL action = 3D: [size_mult, sl_mult, tp_mult]
# - accept/reject is NOT learned; it comes from RiskManager via intent_provider -> env._get_intent()
#   and is applied inside RiskEnv.step() as a hard safety gate.
#
# DQN Implementation:
# - DQN requires discrete actions, so we discretize the continuous action space
# - Each discrete action maps to a combination of [size_mult, sl_mult, tp_mult]
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
import numpy as np

# Gymnasium is preferred for SB3>=2.x
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    gym = None
    spaces = None

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.dqn.policies import DQNPolicy
except Exception:
    DQN = None
    DummyVecEnv = None
    BaseCallback = object
    DQNPolicy = None


# ============================================================
# Config
# ============================================================

@dataclass
class RiskPolicyDQNConfig:
    # Algorithm
    algo: str = "DQN"
    seed: int = 42
    total_timesteps: int = 100_000
    
    # Network architecture
    net_arch: List[int] = None  # For DQN, this will be used in policy_kwargs
    
    # DQN hyperparameters
    dqn_learning_rate: float = 1e-4
    dqn_gamma: float = 0.99
    dqn_buffer_size: int = 100_000
    dqn_batch_size: int = 32
    dqn_learning_starts: int = 1000
    dqn_train_freq: int = 4
    dqn_target_update_interval: int = 1000
    dqn_exploration_fraction: float = 0.1
    dqn_exploration_initial_eps: float = 1.0
    dqn_exploration_final_eps: float = 0.05
    dqn_tau: float = 1.0  # Hard update by default
    
    # Action discretization
    # Number of bins for each action dimension
    size_mult_bins: int = 5  # e.g., [0.0, 0.25, 0.5, 0.75, 1.0]
    sl_mult_bins: int = 5    # e.g., [0.5, 0.875, 1.25, 1.625, 2.0]
    tp_mult_bins: int = 5    # e.g., [0.5, 0.875, 1.25, 1.625, 2.0]
    
    # Eval
    verbose: int = 1
    eval_every_steps: int = 10_000
    eval_episodes: int = 3
    
    # Action bounds for the RL risk controller (multipliers)
    size_mult_min: float = 0.0
    size_mult_max: float = 1.0
    sl_mult_min: float = 0.5
    sl_mult_max: float = 2.0
    tp_mult_min: float = 0.5
    tp_mult_max: float = 2.0
    
    def __post_init__(self):
        if self.net_arch is None:
            self.net_arch = [64, 64]


# ============================================================
# Action Discretization
# ============================================================

class ActionDiscretizer:
    """Discretizes continuous 3D actions into discrete action indices."""
    
    def __init__(self, cfg: RiskPolicyDQNConfig):
        self.cfg = cfg
        
        # Create bins for each dimension
        self.size_mult_values = np.linspace(
            cfg.size_mult_min, cfg.size_mult_max, cfg.size_mult_bins
        )
        self.sl_mult_values = np.linspace(
            cfg.sl_mult_min, cfg.sl_mult_max, cfg.sl_mult_bins
        )
        self.tp_mult_values = np.linspace(
            cfg.tp_mult_min, cfg.tp_mult_max, cfg.tp_mult_bins
        )
        
        # Total number of discrete actions
        self.n_actions = (
            cfg.size_mult_bins * cfg.sl_mult_bins * cfg.tp_mult_bins
        )
        
        # Precompute all action combinations
        self.action_map = []
        for size_idx in range(cfg.size_mult_bins):
            for sl_idx in range(cfg.sl_mult_bins):
                for tp_idx in range(cfg.tp_mult_bins):
                    self.action_map.append([
                        self.size_mult_values[size_idx],
                        self.sl_mult_values[sl_idx],
                        self.tp_mult_values[tp_idx],
                    ])
        self.action_map = np.array(self.action_map, dtype=np.float32)
    
    def discrete_to_continuous(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action index to continuous 3D action."""
        if discrete_action < 0 or discrete_action >= self.n_actions:
            raise ValueError(
                f"Discrete action {discrete_action} out of range [0, {self.n_actions})"
            )
        return self.action_map[discrete_action].copy()
    
    def continuous_to_discrete(self, continuous_action: np.ndarray) -> int:
        """Convert continuous 3D action to nearest discrete action index."""
        size_idx = np.argmin(np.abs(self.size_mult_values - continuous_action[0]))
        sl_idx = np.argmin(np.abs(self.sl_mult_values - continuous_action[1]))
        tp_idx = np.argmin(np.abs(self.tp_mult_values - continuous_action[2]))
        
        return (
            size_idx * self.cfg.sl_mult_bins * self.cfg.tp_mult_bins +
            sl_idx * self.cfg.tp_mult_bins +
            tp_idx
        )


# ============================================================
# Gym wrapper with discrete actions
# ============================================================

class RiskEnvGymWrapperDQN(gym.Env):
    """Wraps the custom RiskEnv to a Gymnasium API with discrete actions for DQN.
    
    Important:
    - This wrapper discretizes the continuous action space
    - accept/reject must be provided by RiskManager through RiskEnv.intent_provider.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, env: Any, cfg: RiskPolicyDQNConfig):
        if gym is None or spaces is None:
            raise ImportError(
                "gymnasium is required for RiskEnvGymWrapperDQN. Please install gymnasium."
            )
        super().__init__()
        self.env = env
        self.cfg = cfg
        self.discretizer = ActionDiscretizer(cfg)
        
        # Discrete action space
        self.action_space = spaces.Discrete(self.discretizer.n_actions)
        
        # Observation space
        obs = self._reset_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
    
    def _reset_obs(self) -> np.ndarray:
        """Reset and get initial observation."""
        warmup_start = 100
        
        try:
            if hasattr(self.env, 'tickers'):
                n_tickers = len(self.env.tickers)
                ticker_idx = (getattr(self.env, 't', 0) % n_tickers) if hasattr(self.env, 't') else 0
            else:
                ticker_idx = 0
            
            out = self.env.reset(start_index=warmup_start, idx_ticker=ticker_idx)
        except (TypeError, AttributeError):
            out = self.env.reset()
        
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
        else:
            obs = out
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        
        # Sanitize observation
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(
                f"Observation contains NaN/Inf after reset. "
                f"Obs shape: {obs.shape}, NaN count: {np.sum(np.isnan(obs))}, "
                f"Inf count: {np.sum(np.isinf(obs))}"
            )
        
        return obs
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs = self._reset_obs()
        
        if not np.all(np.isfinite(obs)):
            raise RuntimeError(f"Observation contains NaN/Inf after reset: {obs}")
        
        return obs, {}
    
    def step(self, discrete_action: int):
        """Step with discrete action."""
        # Convert discrete action to continuous
        continuous_action = self.discretizer.discrete_to_continuous(discrete_action)
        
        # Step environment with continuous action
        obs, reward, done, info = self.env.step(continuous_action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        
        # Sanitize observation
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Sanitize reward
        reward = float(reward)
        if not np.isfinite(reward):
            reward = 0.0
        
        terminated = bool(done)
        truncated = False
        
        # Add continuous action to info for logging
        info['continuous_action'] = continuous_action.tolist()
        info['discrete_action'] = int(discrete_action)
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass


# ============================================================
# Eval callback
# ============================================================

class PeriodicEvalCallbackDQN(BaseCallback):
    """Callback for periodic evaluation during DQN training."""
    
    def __init__(
        self,
        eval_env,
        eval_every_steps=10_000,
        n_episodes=3,
        verbose=1,
        wandb_logger=None
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every_steps = int(eval_every_steps)
        self.n_episodes = int(n_episodes)
        self.wandb_logger = wandb_logger
    
    def _on_step(self) -> bool:
        if self.eval_every_steps > 0 and (self.n_calls % self.eval_every_steps == 0):
            rewards = []
            episode_lengths = []
            
            for _ in range(self.n_episodes):
                reset_out = self.eval_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                
                done = False
                total_reward = 0.0
                steps = 0
                
                while not done:
                    a, _ = self.model.predict(obs, deterministic=True)
                    
                    step_out = self.eval_env.step(a)
                    if len(step_out) == 5:
                        obs, r, terminated, truncated, _info = step_out
                        done = bool(terminated) or bool(truncated)
                    else:
                        obs, r, done, _info = step_out
                        done = bool(done)
                    
                    total_reward += float(r)
                    steps += 1
                    
                    if steps >= 1000:  # Safety limit
                        break
                
                rewards.append(total_reward)
                episode_lengths.append(steps)
            
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_length = np.mean(episode_lengths)
            
            if self.verbose:
                print(
                    f"[Eval @ step {self.n_calls}] "
                    f"mean_reward={mean_reward:.4f} (±{std_reward:.4f}), "
                    f"mean_length={mean_length:.1f}"
                )
            
            # Log to wandb if available
            if self.wandb_logger is not None:
                try:
                    self.wandb_logger.log({
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                        "eval/mean_episode_length": mean_length,
                        "train/global_step": self.n_calls,
                    })
                except Exception:
                    pass  # Silently fail if wandb not properly initialized
        
        return True


# ============================================================
# MAIN POLICY
# ============================================================

class RiskPolicyDQN:
    """DQN-based risk policy."""
    
    def __init__(self, cfg: RiskPolicyDQNConfig):
        self.cfg = cfg
        self.model = None
        self.env = None
        self.discretizer = ActionDiscretizer(cfg)
    
    def build_model(self, env):
        """Build DQN model."""
        if DQN is None:
            raise ImportError("stable-baselines3 is required to build the DQN model.")
        
        # Policy kwargs with network architecture
        policy_kwargs = dict(
            net_arch=self.cfg.net_arch,
        )
        
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=self.cfg.dqn_learning_rate,
            gamma=self.cfg.dqn_gamma,
            buffer_size=self.cfg.dqn_buffer_size,
            batch_size=self.cfg.dqn_batch_size,
            learning_starts=self.cfg.dqn_learning_starts,
            train_freq=self.cfg.dqn_train_freq,
            target_update_interval=self.cfg.dqn_target_update_interval,
            exploration_fraction=self.cfg.dqn_exploration_fraction,
            exploration_initial_eps=self.cfg.dqn_exploration_initial_eps,
            exploration_final_eps=self.cfg.dqn_exploration_final_eps,
            tau=self.cfg.dqn_tau,
            policy_kwargs=policy_kwargs,
            verbose=self.cfg.verbose,
            seed=self.cfg.seed,
        )
    
    def train(self, raw_env: Any, wandb_logger=None):
        """Train DQN on RiskEnv using Gym wrapper."""
        
        if DummyVecEnv is None:
            raise ImportError(
                "stable-baselines3 (with vec_env) is required for training."
            )
        
        # Ensure environment has proper warmup
        warmup_start = 100
        raw_env.reset(start_index=warmup_start, idx_ticker=0)
        
        # Training env
        train_env = DummyVecEnv([lambda: RiskEnvGymWrapperDQN(raw_env, self.cfg)])
        
        # Build model
        self.build_model(train_env)
        
        # Eval env
        eval_env = DummyVecEnv([lambda: RiskEnvGymWrapperDQN(raw_env, self.cfg)])
        
        # Callback with wandb logging
        cb = PeriodicEvalCallbackDQN(
            eval_env,
            self.cfg.eval_every_steps,
            self.cfg.eval_episodes,
            self.cfg.verbose,
            wandb_logger=wandb_logger,
        )
        
        # Train with error handling
        try:
            self.model.learn(
                total_timesteps=self.cfg.total_timesteps,
                callback=cb,
            )
        except RuntimeError as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                raise RuntimeError(
                    f"NaN/Inf detected during training. This usually indicates:\n"
                    f"  1. Insufficient warmup data in environment\n"
                    f"  2. Invalid values in risk features\n"
                    f"  3. Extreme reward values\n"
                    f"Original error: {e}"
                )
            raise
        
        self.env = train_env
        return self.model
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action given observation."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        discrete_action, state = self.model.predict(obs, deterministic=deterministic)
        
        # Convert to continuous action
        continuous_action = self.discretizer.discrete_to_continuous(int(discrete_action))
        
        return continuous_action, state
    
    def save(self, path: str):
        """Save the model."""
        if self.model is None:
            raise RuntimeError("Model not built. Cannot save.")
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, cfg: RiskPolicyDQNConfig):
        """Load a saved model."""
        if DQN is None:
            raise ImportError("stable-baselines3 is required to load the model.")
        
        policy = cls(cfg)
        policy.model = DQN.load(path)
        return policy


# ============================================================
# Fallback random rollout
# ============================================================

def random_rollout_dqn(raw_env, cfg: RiskPolicyDQNConfig, n_steps=50, seed=0):
    """Random rollout for testing."""
    rng = np.random.default_rng(seed)
    discretizer = ActionDiscretizer(cfg)
    env = RiskEnvGymWrapperDQN(raw_env, cfg)
    obs, _ = env.reset()
    total = 0.0
    
    for i in range(n_steps):
        # Random discrete action
        discrete_action = rng.integers(0, discretizer.n_actions)
        step_out = env.step(discrete_action)
        
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
