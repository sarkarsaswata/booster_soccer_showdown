import numpy as np
import torch.nn.functional as F
from sai_rl import SAIClient
from training_scripts.ddpg import DDPG_FF
from training_scripts.training import training_loop

"""
Import-safe training entrypoint for Booster Soccer Showdown (DDPG).

This file avoids creating the SAI client or environment at import time,
so you can import Preprocessor and factories without triggering network calls.
Use main() to actually run training/eval which requires valid SAI credentials.
"""


class Preprocessor:
    def get_task_onehot(self, info):
        if "task_index" in info:
            return info["task_index"]
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        # q expected as (N, 4) -> [x, y, z, w]; v as (3,) or broadcastable
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1, 1) * 2.0)
        return a - b + c

    def modify_state(self, obs, info):
        # Ensure batch dimension
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)

        # Expand info fields to batch if needed
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis=0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis=0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis=0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis=0)
            info["goal_team_0_rel_robot"] = np.expand_dims(info["goal_team_0_rel_robot"], axis=0)
            info["goal_team_1_rel_robot"] = np.expand_dims(info["goal_team_1_rel_robot"], axis=0)
            info["goal_team_0_rel_ball"] = np.expand_dims(info["goal_team_0_rel_ball"], axis=0)
            info["goal_team_1_rel_ball"] = np.expand_dims(info["goal_team_1_rel_ball"], axis=0)
            info["ball_xpos_rel_robot"] = np.expand_dims(info["ball_xpos_rel_robot"], axis=0)
            info["ball_velp_rel_robot"] = np.expand_dims(info["ball_velp_rel_robot"], axis=0)
            info["ball_velr_rel_robot"] = np.expand_dims(info["ball_velr_rel_robot"], axis=0)
            info["player_team"] = np.expand_dims(info["player_team"], axis=0)
            info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_xpos_rel_robot"], axis=0)
            info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_velp_rel_robot"], axis=0)
            info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_xpos_rel_robot"], axis=0)
            info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_velp_rel_robot"], axis=0)
            info["target_xpos_rel_robot"] = np.expand_dims(info["target_xpos_rel_robot"], axis=0)
            info["target_velp_rel_robot"] = np.expand_dims(info["target_velp_rel_robot"], axis=0)
            info["defender_xpos"] = np.expand_dims(info["defender_xpos"], axis=0)

        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))

        obs = np.hstack(
            (
                robot_qpos,
                robot_qvel,
                project_gravity,
                base_ang_vel,
                info["robot_accelerometer"],
                info["robot_velocimeter"],
                info["goal_team_0_rel_robot"],
                info["goal_team_1_rel_robot"],
                info["goal_team_0_rel_ball"],
                info["goal_team_1_rel_ball"],
                info["ball_xpos_rel_robot"],
                info["ball_velp_rel_robot"],
                info["ball_velr_rel_robot"],
                info["player_team"],
                info["goalkeeper_team_0_xpos_rel_robot"],
                info["goalkeeper_team_0_velp_rel_robot"],
                info["goalkeeper_team_1_xpos_rel_robot"],
                info["goalkeeper_team_1_velp_rel_robot"],
                info["target_xpos_rel_robot"],
                info["target_velp_rel_robot"],
                info["defender_xpos"],
                task_onehot,
            )
        )

        return obs


def make_env(comp_id: str = "lower-t1-penalty-kick-goalie"):
    """Create an SAI environment and return (env, sai_client)."""
    sai = SAIClient(comp_id=comp_id)
    env = sai.make_env()
    return env, sai


def create_model(n_features: int, action_space, lr: float = 1e-4):
    """Factory for the DDPG model with the default architecture."""
    return DDPG_FF(
        n_features=n_features,
        action_space=action_space,
        neurons=[24, 12, 6],
        activation_function=F.relu,
        learning_rate=lr,
    )


def action_function_factory(action_space):
    """Create the action mapping function from [-1,1] to env bounds."""

    def map_action(policy):
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (
            expected_bounds[1] - expected_bounds[0]
        )
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        return action_space.low + (action_space.high - action_space.low) * bounded_percent

    return map_action


def main(train_timesteps: int = 1000, comp_id: str = "lower-t1-penalty-kick-goalie"):
    # Create env and client (requires valid SAI credentials)
    env, sai = make_env(comp_id)

    # Preprocessed feature size expected by this Preprocessor setup
    n_features = 87

    # Build model and action mapper
    model = create_model(n_features=n_features, action_space=env.action_space)
    action_function = action_function_factory(env.action_space)

    # Train
    training_loop(env, model, action_function, Preprocessor, timesteps=train_timesteps)

    # Watch and benchmark
    sai.watch(model, action_function, Preprocessor)
    sai.benchmark(model, action_function, Preprocessor)


if __name__ == "__main__":
    main()
