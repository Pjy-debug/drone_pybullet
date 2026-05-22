import os
import sys
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
root_root = os.path.dirname(project_root)
for path in [project_root, root_root]:
    if path not in sys.path:
        sys.path.append(path)

from envs.GlobalPlannerAviary import GlobalPlannerAviary
from utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.AStarPlanner import AStarPlanner
from gym_pybullet_drones.utils.utils import sync

MODEL_PATH = r"C:\Users\86188\Desktop\Drone\drone_pybullet-main\results\save-05.22.2026_14.17.39\best_model.zip"
GUI = True
RECORD_VIDEO = False
NUM_RL_EPISODES = 300

CUSTOM_START = np.array([0.0, 0.0, 0.0])
CUSTOM_GOAL  = np.array([3.5, 0.0, 1.2])
CUSTOM_OBSTACLES = [
    (np.array([1.0, 0.2, 1.0]), 0.35),
    (np.array([2.2, -0.3, 1.3]), 0.4),
    (np.array([1.5, 1.5, 0.8]), 0.3),
]

PID_EPISODE_SEC = 40
WAYPOINT_SPACING = 0.5
ARRIVAL_RADIUS = 0.12
GOAL_TOLERANCE = 0.15
A_STAR_SAFETY_MARGIN = 0.2

def draw_planning_scene(client, start, goal, waypoints, obstacle_list):
    s, g = start, goal
    p.addUserDebugLine(s + [0.15, 0, 0], s - [0.15, 0, 0], [0, 1, 0], 2, physicsClientId=client)
    p.addUserDebugLine(s + [0, 0.15, 0], s - [0, 0.15, 0], [0, 1, 0], 2, physicsClientId=client)
    p.addUserDebugText("START", s + [0, 0, 0.25], [0, 1, 0], 1.2, physicsClientId=client)

    p.addUserDebugLine(g + [0.15, 0, 0], g - [0.15, 0, 0], [0, 0, 1], 2, physicsClientId=client)
    p.addUserDebugLine(g + [0, 0.15, 0], g - [0, 0.15, 0], [0, 0, 1], 2, physicsClientId=client)
    p.addUserDebugText("GOAL", g + [0, 0, 0.25], [0, 0, 1], 1.5, physicsClientId=client)

    path_pts = np.vstack([start, waypoints])
    for i in range(len(path_pts) - 1):
        p.addUserDebugLine(path_pts[i], path_pts[i+1], [1, 1, 0], 1.5, physicsClientId=client)

    for i, wp in enumerate(waypoints):
        p.addUserDebugLine(wp + [0.08, 0, 0], wp - [0.08, 0, 0], [1, 1, 0], 1, physicsClientId=client)
        p.addUserDebugLine(wp + [0, 0.08, 0], wp - [0, 0.08, 0], [1, 1, 0], 1, physicsClientId=client)
        p.addUserDebugText(f"wp{i}", wp + [0, 0, 0.08], [1, 1, 0], 0.8, physicsClientId=client)

    for i, (center, radius) in enumerate(obstacle_list):
        col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
        vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.3], physicsClientId=client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
                          basePosition=center.tolist(), physicsClientId=client)
        p.addUserDebugText(f"obs{i}", center + np.array([0, 0, radius + 0.1]), [1, 0, 0], 0.8, physicsClientId=client)

def test_rl():
    print("\n========== RL Policy Test ==========")
    env = GlobalPlannerAviary(
        gui=GUI, start=CUSTOM_START, goal=CUSTOM_GOAL, obstacles=CUSTOM_OBSTACLES,
        obs=ObservationType.KIN, act=ActionType.PID, record=RECORD_VIDEO
    )
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return None, None

    model = PPO.load(MODEL_PATH)
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ), num_drones=1)

    success_count = 0
    total_steps = 0
    for ep in range(NUM_RL_EPISODES):
        obs, info = env.reset()
        start = time.time()
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            obs_flat = obs.squeeze()
            action_flat = action.squeeze()
            logger.log(drone=0, timestamp=step / env.CTRL_FREQ,
                       state=np.hstack([obs_flat[0:3], np.zeros(4), obs_flat[3:15], action_flat]),
                       control=np.zeros(12))
            if GUI:
                env.render()
                elapsed = time.time() - start
                if step > 0:
                    time_to_wait = env.CTRL_TIMESTEP - elapsed / step
                    if time_to_wait > 0:
                        time.sleep(time_to_wait)
            step += 1

        if info.get('is_success'):
            success_count += 1
            print(f"Episode {ep+1}: success, steps={step}")
        else:
            print(f"Episode {ep+1}: failure, steps={step}")
        total_steps += step

    env.close()
    logger.plot()
    sr = success_count / NUM_RL_EPISODES * 100
    avg_steps = total_steps / NUM_RL_EPISODES
    print(f"RL Success Rate: {sr:.1f}%, Avg Steps: {avg_steps:.1f}")
    return sr, avg_steps

def test_pid():
    print("\n========== PID Baseline ==========")
    planner = AStarPlanner(grid_size=0.2, x_range=(-1.0, 5.0), y_range=(-3.0, 3.0),
                           z_range=(0.1, 2.5), safety_margin=A_STAR_SAFETY_MARGIN)
    raw_path = planner.plan(CUSTOM_START, CUSTOM_GOAL, CUSTOM_OBSTACLES)
    if raw_path is None:
        raw_path = np.stack([CUSTOM_START, CUSTOM_GOAL], axis=0)
    waypoints = AStarPlanner.resample_path(raw_path, spacing=WAYPOINT_SPACING)
    if len(waypoints) > 1:
        waypoints = waypoints[1:]
    waypoints = waypoints.astype(np.float32)
    print(f"Waypoints: {len(waypoints)}")

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=np.array([CUSTOM_START]), initial_rpys=np.array([[0, 0, 0]]),
                     physics=Physics.PYB, pyb_freq=240, ctrl_freq=48, gui=GUI)
    client = env.getPyBulletClient()
    draw_planning_scene(client, CUSTOM_START, CUSTOM_GOAL, waypoints, CUSTOM_OBSTACLES)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    logger = Logger(logging_freq_hz=48, num_drones=1, output_folder="results_pid_baseline")

    action = np.zeros((1, 4))
    wp_index = 0
    max_steps = int(PID_EPISODE_SEC * 48)
    prev_pos = CUSTOM_START.copy()
    final_pos = CUSTOM_START.copy()
    steps_taken = 0
    start_time = time.time()

    for i in range(max_steps):
        obs, _, _, _, _ = env.step(action)
        state = obs[0]
        pos = state[0:3]
        final_pos = pos
        target = waypoints[wp_index]

        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=state,
            target_pos=target, target_rpy=np.array([0, 0, 0]))

        p.addUserDebugLine(prev_pos, pos, [1, 1, 1], 1.2, 0, physicsClientId=client)
        prev_pos = pos.copy()

        dist = np.linalg.norm(pos - target)
        if dist < ARRIVAL_RADIUS:
            print(f"Reached wp{wp_index}, dist={dist:.3f}")
            if wp_index < len(waypoints) - 1:
                wp_index += 1
            else:
                steps_taken = i + 1
                break

        logger.log(drone=0, timestamp=i / 48, state=state,
                   control=np.hstack([target, np.zeros(9)]))
        env.render()
        if GUI:
            sync(i, start_time, env.CTRL_TIMESTEP)
        steps_taken = i + 1

    env.close()
    logger.save()
    logger.save_as_csv("pid_trajectory")
    logger.plot()

    final_dist = np.linalg.norm(final_pos - CUSTOM_GOAL)
    success = final_dist < GOAL_TOLERANCE
    print(f"PID success: {success}, steps: {steps_taken}, final dist: {final_dist:.3f}m")
    return success, steps_taken, final_dist

if __name__ == "__main__":
    rl_sr, rl_avg_steps = test_rl()
    pid_success, pid_steps, pid_final_dist = test_pid()
    print("\n==================== Summary ====================")
    print(f"RL  Success Rate: {rl_sr:.1f}%  Avg Steps: {rl_avg_steps:.1f}")
    print(f"PID Success: {pid_success}, Steps: {pid_steps}, Final Dist: {pid_final_dist:.3f}m")
    print("==================================================")