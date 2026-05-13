"""基于 GlobalPlannerAviary 的强化学习训练脚本。

该脚本使用 PPO 算法训练无人机按照全局规划的路径点进行飞行。
"""
import os
import sys

# 获取当前脚本所在目录的上一级目录（即项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
root_root = os.path.dirname(project_root)
# 将项目根目录添加到系统路径中
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
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

from utils.Logger import Logger
# 导入自定义的全局规划环境
from envs.GlobalPlannerAviary import GlobalPlannerAviary 
from utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

from tqdm import tqdm

# --- 默认参数配置 ---
DEFAULT_GUI = False      # 训练时关闭 GUI (快很多); 测试阶段自动打开
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_TIMESTEPS = int(3e6)   # 总训练步数

DEFAULT_OBS = ObservationType.KIN # 运动学观测
DEFAULT_ACT = ActionType.PID    # 推荐使用速度控制 ('pid') 以实现更好的路径追踪
DEFAULT_AGENTS = 1
DEFAULT_MA = False

# --- wandb ---
WANDB_PROJECT = 'drone-rl-avoidance'
WANDB_ENTITY = None     # 填你的 wandb entity (可选)
WANDB_MODE = 'online'   # 'online' / 'offline' / 'disabled'

# --- 简化场景: start / goal 在两个小球形区域内变化, 障碍生成在两点连线附近 ---
START_CENTER   = (0.0, 0.0, 1.0)
GOAL_CENTER    = (3.5, 0.0, 1.0)
REGION_RADIUS  = 0.3   # start / goal 球形扰动半径
NUM_OBSTACLES  = 2     # 沿连线放置的障碍数

from stable_baselines3.common.callbacks import BaseCallback

class IterationCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.iteration = 0

    def _on_step(self) -> bool:
        return True  # 必须有

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        print(f"[INFO] 当前训练轮次 (PPO update): {self.iteration}")

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

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, timesteps=DEFAULT_TIMESTEPS, resume=None):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
    print('filename is ', filename)

    # --- wandb 初始化 ---
    run_name = os.path.basename(filename)
    wandb_config = dict(
        algo='PPO',
        timesteps=timesteps,
        start_center=START_CENTER,
        goal_center=GOAL_CENTER,
        region_radius=REGION_RADIUS,
        num_obstacles=NUM_OBSTACLES,
        obs=str(DEFAULT_OBS),
        act=str(DEFAULT_ACT),
        n_envs=72,
    )
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=wandb_config,
        sync_tensorboard=True,   # 让 SB3 写的 TB scalar 自动镜像到 wandb
        save_code=True,
        mode=WANDB_MODE,
        dir=filename,
    )
    # 封装环境初始化参数
    env_kwargs = dict(
        simple_scene=True,
        start_center=START_CENTER,
        goal_center=GOAL_CENTER,
        region_radius=REGION_RADIUS,
        num_obstacles=NUM_OBSTACLES,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
    )

    if multiagent:
        env_kwargs["num_drones"] = DEFAULT_AGENTS
    train_env = make_vec_env(GlobalPlannerAviary,
                             env_kwargs=env_kwargs,
                             n_envs=72,
                             seed=0)
    # 评测环境不开 GUI, 以免每次 eval 都渲染拖慢训练
    eval_env = GlobalPlannerAviary(**env_kwargs, gui=DEFAULT_GUI)

    #### 打印空间信息 ########################################
    # print('[INFO] Action space:', train_env.action_space)
    # print('[INFO] Observation space:', train_env.observation_space)

    #### 训练模型 #############################################
    # learning_rate=3e-4 为恒定 (constant LR), SB3 里传数字默认就是 constant.
    # tensorboard_log: SB3 写 TB scalar 供 wandb sync_tensorboard 镜像, 不需要手动看 TB.
    tb_dir = os.path.join(filename, 'tb')
    if resume is not None and os.path.isfile(resume):
        print(f'[INFO] resume 训练, 从 {resume} 加载模型 + optimizer state')
        model = PPO.load(
            resume,
            env=train_env,
            device='auto',
            tensorboard_log=tb_dir,
            # 覆盖部分超参 (以防旧 ckpt 超参不合适)
            custom_objects={
                'learning_rate': 3e-4,
                'clip_range': 0.2,
                'ent_coef': 0.005,
            },
        )
    else:
        model = PPO('MlpPolicy',
                    train_env,
                    tensorboard_log=tb_dir,
                    learning_rate=3e-4,
                    n_steps=256,
                    batch_size=4096,
                    n_epochs=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.005,
                    verbose=1,
                    device='auto')
    #### 目标奖励阈值 (根据路径点数量调整) #######################
    # 路径追踪任务通常需要更长的时间，target_reward 需根据实际奖励曲线调整
    target_reward = 1000. if not multiagent else 2000.
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=2000,
                                 n_eval_episodes=3,
                                 deterministic=True,
                                 render=False)

    iteration_callback = IterationCounterCallback()
    total_timesteps = timesteps if local else int(1e4)
    tqdm_callback = TqdmCallback(total_timesteps)
    wandb_callback = WandbCallback(
        model_save_path=filename,
        model_save_freq=20000,
        gradient_save_freq=0,
        verbose=2,
    )
    # 定期 checkpoint (包含 policy + optimizer state, 可用于续训)
    ckpt_dir = os.path.join(filename, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // 72, 1),  # 按总步数 ~50k 存一次
        save_path=ckpt_dir,
        name_prefix='ppo_ckpt',
        save_replay_buffer=False,        # PPO 是 on-policy, 无 replay buffer
        save_vecnormalize=False,
    )
    model.learn(total_timesteps=total_timesteps,
                callback=[eval_callback, iteration_callback, tqdm_callback,
                          wandb_callback, checkpoint_callback],
                log_interval=2,
                reset_num_timesteps=(resume is None))

    #### 保存模型 #############################################
    model.save(filename+'/final_model.zip')

    #### 加载最佳模型进行测试 ###################################
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        path = filename+'/final_model.zip'
    model = PPO.load(path)

    #### 测试与演示 ###########################################
    test_env = GlobalPlannerAviary(gui=False,
                                   simple_scene=True,
                                   start_center=START_CENTER,
                                   goal_center=GOAL_CENTER,
                                   region_radius=REGION_RADIUS,
                                   num_obstacles=NUM_OBSTACLES,
                                   obs=DEFAULT_OBS,
                                   act=DEFAULT_ACT,
                                   record=record_video,
                                   num_drones=DEFAULT_AGENTS if multiagent else 1)

    test_env_nogui = GlobalPlannerAviary(gui=False,
                                         simple_scene=True,
                                         start_center=START_CENTER,
                                         goal_center=GOAL_CENTER,
                                         region_radius=REGION_RADIUS,
                                         num_obstacles=NUM_OBSTACLES,
                                         obs=DEFAULT_OBS,
                                         act=DEFAULT_ACT,
                                         num_drones=DEFAULT_AGENTS if multiagent else 1)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=filename,
                colab=colab)

    mean_reward, std_reward = evaluate_policy(model, test_env_nogui, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42)
    start = time.time()
    
    # 路径追踪可能需要比普通悬停更长的步数
    for i in range((test_env.EPISODE_LEN_SEC + 5) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        
        if DEFAULT_OBS == ObservationType.KIN:
            # 注意：GlobalPlannerAviary 的 obs 包含了 12维状态 + Buffer + 3维目标
            # 这里的切片处理需要与环境中的 _computeObs 保持一致
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                        control=np.zeros(12)
                        )
        
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            obs, info = test_env.reset(seed=42)
            
    test_env.close()
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()
    logger.save_as_csv()
    logger.save()

    # 关闭 wandb
    try:
        wandb_run.finish()
    except Exception:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning with Global Planning')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool)
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool)
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool)
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--timesteps',          default=DEFAULT_TIMESTEPS,     type=int)
    parser.add_argument('--resume',             default=None,                  type=str,
                        help='从已保存的 .zip 模型续训 (包含 policy + optimizer state)')
    ARGS = parser.parse_args()

    run(**vars(ARGS))