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
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from utils.Logger import Logger
# 导入自定义的全局规划环境
from envs.GlobalPlannerAviary import GlobalPlannerAviary 
from utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType

from tqdm import tqdm

# --- 默认参数配置 ---
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType.KIN # 运动学观测
DEFAULT_ACT = ActionType.PID    # 推荐使用速度控制 ('pid') 以实现更好的路径追踪
DEFAULT_AGENTS = 1
DEFAULT_MA = False

# --- 定义全局路径点 (Global Waypoints) ---
# 无人机将依次通过这些坐标点
CUSTOM_WAYPOINTS = np.array([
    [1.0, 1.0, 1.0],
    [2.0, 0.0, 1.2],
    [3.0, 2.0, 1.0],
    [0.0, 0.0, 1.5]
])

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

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    # 封装环境初始化参数
    env_kwargs = dict(
        waypoints=CUSTOM_WAYPOINTS,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT
    )

    if not multiagent:
        train_env = make_vec_env(GlobalPlannerAviary,
                                 env_kwargs=env_kwargs,
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = GlobalPlannerAviary(**env_kwargs, 
                                 gui = DEFAULT_GUI)
    else:
        # 多智能体逻辑，需确保 GlobalPlannerAviary 支持 num_drones 参数
        env_kwargs["num_drones"] = DEFAULT_AGENTS
        train_env = make_vec_env(GlobalPlannerAviary,
                                 env_kwargs=env_kwargs,
                                 n_envs=1,
                                 seed=0
                                 )
        eval_env = GlobalPlannerAviary(**env_kwargs, 
                                 gui = DEFAULT_GUI)

    #### 打印空间信息 ########################################
    # print('[INFO] Action space:', train_env.action_space)
    # print('[INFO] Observation space:', train_env.observation_space)

    #### 训练模型 #############################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                learning_rate=0.0003,      # 学习率
                n_steps=4800,               # 每次更新前收集的数据步数
                batch_size=64,              # 每次优化时的批次大小
                gamma=0.99,                 # 折扣因子（对未来奖励的重视程度）
                gae_lambda=0.95,            # GAE 参数
                clip_range=0.2,             # PPO 裁剪范围
                ent_coef=0.01,              # 熵系数（鼓励探索）
                verbose=1,
                device= 'cuda')
    # model = PPO.load("D:\\homework\\Grade3_2\\drone\\gym-pybullet-drones-main\\results\\save-04.17.2026_12.01.43\\final_model.zip", env= train_env)
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
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=True)

    iteration_callback = IterationCounterCallback()
    total_timesteps = int(2e6)
    tqdm_callback = TqdmCallback(total_timesteps)

    model.learn(total_timesteps=int(2e6) if local else int(1e4), 
                callback=[eval_callback, iteration_callback, tqdm_callback],
                log_interval=100)

    #### 保存模型 #############################################
    model.save(filename+'/final_model.zip')

    #### 加载最佳模型进行测试 ###################################
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        path = filename+'/final_model.zip'
    model = PPO.load(path)

    #### 测试与演示 ###########################################
    test_env = GlobalPlannerAviary(gui=gui,
                                   waypoints=CUSTOM_WAYPOINTS,
                                   obs=DEFAULT_OBS,
                                   act=DEFAULT_ACT,
                                   record=record_video,
                                   num_drones=DEFAULT_AGENTS if multiagent else 1)
    
    test_env_nogui = GlobalPlannerAviary(waypoints=CUSTOM_WAYPOINTS,
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning with Global Planning')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool)
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool)
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool)
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str)
    ARGS = parser.parse_args()

    run(**vars(ARGS))