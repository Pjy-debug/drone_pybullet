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
import numpy as np
import torch
from stable_baselines3 import PPO
from envs.GlobalPlannerAviary import GlobalPlannerAviary
from utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# --- 配置参数 (需与训练时保持一致) ---
MODEL_PATH = "D:\\homework\\Grade3_2\\drone\\gym-pybullet-drones-main\\gym_pybullet_drones\\results\\save-04.24.2026_03.24.42\\best_model.zip"
GUI = True              # 测试时建议打开可视化
RECORD_VIDEO = False
NUM_EPISODES = 3        # 测试的回合数

CUSTOM_START = np.array([0.0, 0.0, 0.5])
CUSTOM_GOAL  = np.array([3.5, 0.0, 1.2])
CUSTOM_OBSTACLES = [
    (np.array([1.0, 0.2, 1.0]), 0.35),
    (np.array([2.2, -0.3, 1.3]), 0.4),
    (np.array([1.5, 1.5, 0.8]), 0.3),
]

def test():
    #### 1. 加载环境 #############################################
    test_env = GlobalPlannerAviary(
        gui=GUI,
        start=CUSTOM_START,
        goal=CUSTOM_GOAL,
        obstacles=CUSTOM_OBSTACLES,
        obs=ObservationType.KIN,
        act=ActionType.PID,
        record=RECORD_VIDEO
    )

    #### 2. 加载训练好的模型 #######################################
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 找不到模型文件: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH)
    print(f"[INFO] 成功加载模型: {MODEL_PATH}")

    #### 3. 初始化数据记录器 #######################################
    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1
    )

    #### 4. 执行测试循环 ###########################################
    for episode in range(NUM_EPISODES):
        obs, info = test_env.reset()
        start_time = time.time()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\n--- 开始第 {episode+1} 回合测试 ---")

        # 运行直到回合结束 (terminated 或 truncated)
        while not done:
            # 模型推理
            action, _states = model.predict(obs, deterministic=True)
            
            # 环境步进
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # 记录数据用于后期绘图 (需匹配你环境的 obs 结构)
            # 假设 obs 前12维是基本运动状态
            obs_flat = obs.squeeze()
            action_flat = action.squeeze()
            logger.log(
                drone=0,
                timestamp=step / test_env.CTRL_FREQ,
                state=np.hstack([obs_flat[0:3], np.zeros(4), obs_flat[3:15], action_flat]),
                control=np.zeros(12)
            )

            # 渲染与同步
            if GUI:
                test_env.render()
                # 控制仿真速度与真实时间同步
                time_to_wait = test_env.CTRL_TIMESTEP - (time.time() - start_time) / (step + 1)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
            
            step += 1

        print(f"回合结束: 步数={step}, 总奖励={episode_reward:.2f}")
        if info.get('is_success'):
            print("结果: [成功] 到达目标点!")
        else:
            print("结果: [失败] 碰撞或超时")

    #### 5. 结果可视化 #############################################
    test_env.close()
    print("\n[INFO] 正在生成分析图表...")
    logger.plot()  # 绘制状态曲线 (XYZ, RPY, Velocity)

if __name__ == "__main__":
    test()