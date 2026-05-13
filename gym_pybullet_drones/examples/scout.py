import os
import sys
import time
import numpy as np

# ========= 路径 =========
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
root_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.append(project_root)

if root_root not in sys.path:
    sys.path.append(root_root)

# ========= 环境 =========
from envs.GlobalPlannerAviary import GlobalPlannerAviary

from gym_pybullet_drones.utils.enums import (
    ObservationType,
    ActionType
)

# ========= 参数 =========
START_CENTER = (0.0, 0.0, 1.0)
GOAL_CENTER  = (3.5, 0.0, 1.0)

REGION_RADIUS = 0.3
NUM_OBSTACLES = 2

OBS = ObservationType.KIN
ACT = ActionType.PID

# ========= 创建环境 =========
env = GlobalPlannerAviary(
    gui=True,
    record=False,

    simple_scene=True,

    start_center=START_CENTER,
    goal_center=GOAL_CENTER,

    region_radius=REGION_RADIUS,
    num_obstacles=NUM_OBSTACLES,

    obs=OBS,
    act=ACT,
)

print("\n========== 环境可视化模式 ==========\n")

EPISODES = 1000

for ep in range(EPISODES):

    print(f"\n========== Episode {ep} ==========\n")

    obs, info = env.reset()

    print("[INFO] reset 完成")

    # ========= 打印调试信息 =========
    try:
        print("[INFO] start_pos:", env.START_POS)
    except:
        pass

    try:
        print("[INFO] goal_pos:", env.goal_pos)
    except:
        pass

    try:
        print("[INFO] obstacles:", env.obstacle_positions)
    except:
        pass

    try:
        print("[INFO] global path:")
        print(env.global_path)
    except:
        pass

    # ========= 静止观察 =========
    for i in range(240 * 20):

        # PID action维度
        action = np.zeros((1, 9), dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        time.sleep(1. / 240.)

        # 每隔几秒打印一次无人机位置
        if i % 240 == 0:
            try:
                drone_pos = obs[0:3]
                print(f"[INFO] drone_pos: {drone_pos}")
            except:
                pass

        if terminated or truncated:
            print("[INFO] episode ended")
            break

    print("\n等待2秒进入下一张地图...\n")

    time.sleep(2)

env.close()