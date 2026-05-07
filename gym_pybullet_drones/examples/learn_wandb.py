"""基于 GlobalPlannerAviary 的强化学习训练脚本（已集成 wandb）"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
root_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)
if root_root not in sys.path:
    sys.path.append(root_root)

import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# ✅ wandb
import wandb
from wandb.integration.sb3 import WandbCallback

from utils.Logger import Logger
from envs.GlobalPlannerAviary import GlobalPlannerAviary 
from utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from tqdm import tqdm

# --- 默认参数 ---
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_TIMESTEPS = int(5e6)

DEFAULT_OBS = ObservationType.KIN
DEFAULT_ACT = ActionType.PID
DEFAULT_AGENTS = 1
DEFAULT_MA = False

CUSTOM_START = np.array([0.0, 0.0, 0.5])
CUSTOM_GOAL  = np.array([3.5, 0.0, 1.2])
CUSTOM_OBSTACLES = [
    (np.array([1.0, 0.2, 1.0]), 0.35),
    (np.array([2.2, -0.3, 1.3]), 0.4),
    (np.array([1.5, 1.5, 0.8]), 0.3),
]

# ================= Callback =================

class IterationCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.iteration = 0

    def _on_step(self) -> bool:
        return True  # ⭐必须有，否则会报错

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        print(f"[INFO] PPO iteration: {self.iteration}")

        import wandb
        wandb.log({"train/ppo_iteration": self.iteration})


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)

    def _on_step(self):
        self.pbar.n = self.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.close()

# ================= 主函数 =================

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI,
        plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO,
        local=True, timesteps=DEFAULT_TIMESTEPS):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    os.makedirs(filename, exist_ok=True)

    print('filename is ', filename)

    # ================= wandb 初始化 =================
    run = wandb.init(
        project="ppo-global-planner",
        name=filename.split('/')[-1],
        config={
            "algo": "PPO",
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 1024,
            "gamma": 0.99,
            "env": "GlobalPlannerAviary",
            "n_envs": 72,
            "timesteps": timesteps,
        },
        sync_tensorboard=True,   # ⭐关键
        monitor_gym=True,
        save_code=True
    )

    env_kwargs = dict(
        start=CUSTOM_START,
        goal=CUSTOM_GOAL,
        obstacles=CUSTOM_OBSTACLES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
    )

    if multiagent:
        env_kwargs["num_drones"] = DEFAULT_AGENTS

    train_env = make_vec_env(GlobalPlannerAviary, env_kwargs=env_kwargs, n_envs=72, seed=0)
    eval_env = GlobalPlannerAviary(**env_kwargs, gui=False)

    # ================= PPO =================
    model = PPO(
        'MlpPolicy',
        train_env,
        tensorboard_log=f"{wandb.run.dir}/tb",  # ⭐关键
        learning_rate=3e-4,
        n_steps=512,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        device='auto'
    )

    target_reward = 1000. if not multiagent else 2000.

    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename,
        log_path=filename,
        eval_freq=2000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    iteration_callback = IterationCounterCallback()
    total_timesteps = timesteps if local else int(1e4)
    tqdm_callback = TqdmCallback(total_timesteps)

    # ================= 训练 =================
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            eval_callback,
            iteration_callback,
            tqdm_callback,
            WandbCallback(
                gradient_save_freq=1000,
                model_save_path=f"{wandb.run.dir}/models",
                verbose=2,
            )
        ],
        log_interval=10
    )

    # ================= 保存 =================
    model.save(filename+'/final_model.zip')

    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        path = filename+'/final_model.zip'

    model = PPO.load(path)

    # ================= 评估 =================
    test_env_nogui = GlobalPlannerAviary(
        gui=False,
        start=CUSTOM_START,
        goal=CUSTOM_GOAL,
        obstacles=CUSTOM_OBSTACLES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        num_drones=DEFAULT_AGENTS if multiagent else 1
    )

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)

    print("\nMean reward ", mean_reward, " +- ", std_reward)

    # ✅ 记录最终结果
    wandb.log({
        "final/mean_reward": mean_reward,
        "final/std_reward": std_reward
    })

    # ================= 关闭 wandb =================
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--timesteps', default=DEFAULT_TIMESTEPS, type=int)

    ARGS = parser.parse_args()
    run(**vars(ARGS))