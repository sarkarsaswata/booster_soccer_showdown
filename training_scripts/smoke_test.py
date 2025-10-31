import sys
import os
import numpy as np
import torch

# Import model and preprocessor definition from training main (now import-safe)
from training_scripts.ddpg import DDPG_FF  # type: ignore
from training_scripts.preprocessor import Preprocessor  # type: ignore

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)


class LightBox:
    """Minimal Box-like space with low/high/shape for action mapping."""

    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class MockEnv:
    """Gymnasium-like mock env that matches the Preprocessor contract.

    - Observation: 24D (12 qpos + 12 qvel)
    - Info dict: contains all keys referenced by Preprocessor with arbitrary shapes
    - Action space: 12D in [-1, 1]
    """

    def __init__(self, action_dim: int = 12, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.action_space = LightBox(low=-np.ones(action_dim), high=np.ones(action_dim))

    def _info(self):
        # Provide minimally valid shapes; values are synthetic but consistent.
        info = {
            "robot_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # [x,y,z,w]
            "robot_gyro": self.rng.normal(0, 0.1, size=(3,)).astype(np.float32),
            "robot_accelerometer": self.rng.normal(0, 0.1, size=(3,)).astype(np.float32),
            "robot_velocimeter": self.rng.normal(0, 0.1, size=(3,)).astype(np.float32),
            "goal_team_0_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goal_team_1_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goal_team_0_rel_ball": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goal_team_1_rel_ball": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "ball_xpos_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "ball_velp_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "ball_velr_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "player_team": np.array([1.0], dtype=np.float32),
            "goalkeeper_team_0_xpos_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goalkeeper_team_0_velp_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goalkeeper_team_1_xpos_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "goalkeeper_team_1_velp_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "target_xpos_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "target_velp_rel_robot": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "defender_xpos": self.rng.normal(0, 1.0, size=(3,)).astype(np.float32),
            "task_index": np.array([1.0, 0.0, 0.0], dtype=np.float32),  # 3-task one-hot
        }
        return info

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs = np.zeros(24, dtype=np.float32)
        # 12 qpos + 12 qvel synthetic state
        obs[:12] = self.rng.normal(0, 0.01, size=(12,)).astype(np.float32)
        obs[12:24] = self.rng.normal(0, 0.01, size=(12,)).astype(np.float32)
        return obs, self._info()

    def step(self, action):
        # Bound action and return a single-step truncated episode
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs = np.zeros(24, dtype=np.float32)
        reward = float(-np.linalg.norm(action) * 0.01)
        terminated = False
        truncated = True  # end immediately (smoke test)
        info = self._info()
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


def main():
    env = MockEnv(action_dim=12)
    pre = Preprocessor()

    print("[SmokeTest] Resetting mock env…")
    obs, info = env.reset()

    # Preprocess
    proc = pre.modify_state(obs, info)
    if proc.ndim == 1:
        proc = np.expand_dims(proc, axis=0)

    n_features = proc.shape[1]
    print(f"[SmokeTest] Preprocessed obs shape: {proc.shape} (n_features={n_features})")

    # Build model consistent with env and preprocessed features
    model = DDPG_FF(
        n_features=n_features,
        action_space=env.action_space,
        neurons=[24, 12, 6],
        activation_function=torch.nn.functional.relu,
        learning_rate=1e-4,
    )

    # Forward pass
    with torch.no_grad():
        out = model(torch.from_numpy(proc))
    print(f"[SmokeTest] Actor output shape: {out.shape}")

    # Map action from [-1,1] to env action space
    expected_bounds = [-1, 1]
    action_percent = (out.numpy() - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    action = env.action_space.low + (env.action_space.high - env.action_space.low) * bounded_percent

    # Step once
    print("[SmokeTest] Stepping mock env once…")
    obs2, reward, terminated, truncated, info2 = env.step(action[0])
    print(f"[SmokeTest] Step ok. reward={float(reward):.3f}, terminated={terminated}, truncated={truncated}")

    env.close()
    print("[SmokeTest] Done.")


if __name__ == "__main__":
    main()
