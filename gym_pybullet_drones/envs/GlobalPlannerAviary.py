"""GlobalPlannerAviary: A* + 障碍物 + RL target 追踪.

关键改动:
    * __init__ 中用 AStarPlanner 在给定 (start, goal, obstacles) 上规划 waypoints,
      作为 RL 追踪的子目标序列.
    * 重写 _actionSpace / _preprocessAction: action 不再裁剪到 [-1,1], 而是直接
      作为世界坐标下的绝对 PID 目标点.
    * 重写 _addObstacles: 把规划时使用的障碍球加到 PyBullet 里, 保证训练和规划
      看到同一套障碍.
    * reward 中加入避障惩罚; GUI 下绘制起点 / 终点 / 障碍 / A* 路径 / 当前目标.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces
import os
import csv
from datetime import datetime

from envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.AStarPlanner import AStarPlanner
from utils.SceneGenerator import SceneGenerator



def default_obstacles():
    """训练 / 评测默认使用的一组固定障碍 (位置, 半径)."""
    return [
        (np.array([1.0, 0.2, 1.0]), 0.35),
        (np.array([2.2, -0.3, 1.3]), 0.4),
        (np.array([1.5, 1.5, 0.8]), 0.3),
    ]


class GlobalPlannerAviary(BaseRLAviary):
    """带 A* 全局规划 + 固定障碍 + RL 子目标追踪的无人机环境."""

    def __init__(self,
                 start: np.ndarray = np.array([0.0, 0.0, 0.5]),
                 goal: np.ndarray = np.array([3.5, 0.0, 1.2]),
                 obstacles=None,
                 waypoint_spacing: float = 0.6,
                 arrival_radius: float = 0.15,
                 workspace_bounds=((-1.0, 5.0), (-3.0, 3.0), (-0.1, 2.5)),
                 act_scale: float = None,   # 预留, 当前未使用
                 collision_penalty: float = 50.0,
                 # --- 简化任务: start / goal 在小球形区域随机, 障碍生成在连线上 ---
                 simple_scene: bool = True,
                 start_center=(0.0, 0.0, 1.0),
                 goal_center=(3.5, 0.0, 1.0),
                 region_radius: float = 0.3,
                 num_obstacles: int = 2,
                 IS_DEBUG: bool = False,
                 task: str = "global",
                 mode = "training",
                 **kwargs):
        self.mode = mode

        self.task = task

        self.IS_DEBUG = IS_DEBUG
        # --- 规划相关参数必须在父类调用前准备好 (父类会调 _addObstacles / _actionSpace) ---
        if self.IS_DEBUG:
            print(f'工作空间限制: x={workspace_bounds[0]}, y={workspace_bounds[1]}, z={workspace_bounds[2]}')
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存场景生成参数, reset 时复用
        self._simple_scene = simple_scene
        self._start_center = tuple(start_center)
        self._goal_center = tuple(goal_center)
        self._region_radius = float(region_radius)
        self._num_obstacles = int(num_obstacles)

        random_start, random_goal, random_obstacles = self._sample_scene(workspace_bounds)
        self.START = np.asarray(random_start, dtype=float)
        self.GOAL = np.asarray(random_goal, dtype=float)
        # 注意: BaseAviary 存在同名 bool 属性 self.OBSTACLES (在 super().__init__ 里赋值),
        # 这里用 OBSTACLE_LIST 避免被父类覆盖.
        self.OBSTACLE_LIST = random_obstacles if random_obstacles is not None else default_obstacles()
        self.ARRIVAL_RADIUS = arrival_radius
        self.COLLISION_PENALTY = collision_penalty
        self.WORKSPACE = workspace_bounds
        self._waypoint_spacing = waypoint_spacing

        # self.target = self.START.copy()

        # --- 跑 A*, 生成 waypoints ---
        self.WAYPOINTS = self._plan_path()

        # 让无人机从 start 起飞
        kwargs.setdefault('initial_xyzs', self.START.reshape(1, 3))

        super().__init__(**kwargs)

        self.EPISODE_LEN_SEC = 15
        self.num_waypoints = self.WAYPOINTS.shape[0]
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)

        # 可视化 id 记录
        self._dbg_item_ids = []
        if self.GUI:
            self._draw_planning_scene()

        self._reset_episode_metrics()



    def _reset_episode_metrics(self):
        self.episode_metrics = {
            # 时间
            "episode_steps": 0,
    
            # 路径
            "path_length": 0.0,
            "straight_distance": np.linalg.norm(self.GOAL - self.START),
    
            # 速度
            "speed_sum": 0.0,
            "speed_sq_sum": 0.0,
    
            # 姿态
            "roll_sq_sum": 0.0,
            "pitch_sq_sum": 0.0,
    
            # waypoint
            "waypoints_reached": 0,
    
            # 障碍
            "min_obstacle_distance": np.inf,
            "collision_count": 0,
    
            # 轨迹质量
            "tracking_error_sum": 0.0,
    
            # 能耗
            "rpm_sum": 0.0,
    
            # 终止原因
            "termination_reason": "unknown",
        }

    def _update_episode_metrics(self):
        """
        每个 step 调用一次
        """
        if not hasattr(self, "_last_positions"):
            self._last_positions = np.array([
                self._getDroneStateVector(i)[0:3]
                for i in range(self.NUM_DRONES)
            ])

        self.episode_metrics["episode_steps"] += 1
    
        for i in range(self.NUM_DRONES):
    
            state = self._getDroneStateVector(i)
    
            pos = state[0:3]
            vel = state[10:13]
            roll, pitch, _ = state[7:10]
    
            # -------------------------
            # 路径长度
            # -------------------------
            if self._last_positions is not None:
                step_dist = np.linalg.norm(
                    pos - self._last_positions[i]
                )
    
                self.episode_metrics["path_length"] += step_dist
    
                self._last_positions[i] = pos.copy()
    
            # -------------------------
            # 速度统计
            # -------------------------
            speed = np.linalg.norm(vel)
    
            self.episode_metrics["speed_sum"] += speed
            self.episode_metrics["speed_sq_sum"] += speed**2
    
            # -------------------------
            # 姿态稳定性
            # -------------------------
            self.episode_metrics["roll_sq_sum"] += roll**2
            self.episode_metrics["pitch_sq_sum"] += pitch**2
    
            # -------------------------
            # waypoint
            # -------------------------
            self.episode_metrics["waypoints_reached"] = int(
                np.sum(self.wp_counters)
            )
    
            # -------------------------
            # 障碍距离
            # -------------------------
            d_obs = self._min_obs_distance(pos)
    
            self.episode_metrics["min_obstacle_distance"] = min(
                self.episode_metrics["min_obstacle_distance"],
                d_obs
            )
    
            if d_obs < 0:
                self.episode_metrics["collision_count"] += 1
    
            # -------------------------
            # tracking error
            # -------------------------
            target_wp = self.WAYPOINTS[self.wp_counters[i]]
    
            tracking_error = np.linalg.norm(
                target_wp - pos
            )
    
            self.episode_metrics["tracking_error_sum"] += tracking_error
    
            # -------------------------
            # rpm
            # -------------------------
            if hasattr(self, "rpm"):
                self.episode_metrics["rpm_sum"] += np.mean(
                    self.rpm[i]
                )

    def _finalize_episode_metrics(self):
        steps = max(
            self.episode_metrics["episode_steps"],
            1
        )
    
        speed_mean = (
            self.episode_metrics["speed_sum"] / steps
        )
    
        speed_var = (
            self.episode_metrics["speed_sq_sum"] / steps
            - speed_mean**2
        )
    
        self.episode_metrics["avg_speed"] = speed_mean
    
        self.episode_metrics["speed_std"] = np.sqrt(
            max(speed_var, 0.0)
        )
    
        self.episode_metrics["avg_roll"] = np.sqrt(
            self.episode_metrics["roll_sq_sum"] / steps
        )
    
        self.episode_metrics["avg_pitch"] = np.sqrt(
            self.episode_metrics["pitch_sq_sum"] / steps
        )
    
        self.episode_metrics["avg_tracking_error"] = (
            self.episode_metrics["tracking_error_sum"] / steps
        )
    
        self.episode_metrics["avg_rpm"] = (
            self.episode_metrics["rpm_sum"] / steps
        )
    
        path_len = self.episode_metrics["path_length"]
    
        straight_len = self.episode_metrics["straight_distance"]
    
        self.episode_metrics["path_efficiency"] = (
            straight_len / max(path_len, 1e-6)
        )

    def _save_episode_metrics(self):
    
        save_dir = "episode_logs"
        os.makedirs(save_dir, exist_ok=True)
    
        csv_path = os.path.join(
            save_dir,
            "metrics"+self.start_time+".csv"
        )
    
        file_exists = os.path.exists(csv_path)
    
        with open(csv_path, "a", newline="") as f:
    
            writer = csv.DictWriter(
                f,
                fieldnames=self.episode_metrics.keys()
            )
    
            if not file_exists:
                writer.writeheader()
    
            writer.writerow(self.episode_metrics)
    
    # -------------------------------------------------------------- scene

    def _sample_scene(self, workspace_bounds):
        """按当前模式随机生成 (start, goal, obstacles)."""
        (xr, yr, zr) = workspace_bounds
        scene_gen = SceneGenerator(x_range=xr, y_range=yr, z_range=zr)
        if self._simple_scene:
            if self.task == 'easy':
                return scene_gen.generate_takeoff()
            elif self.task == 'short':
                return scene_gen.generate_short_distance()
            elif self.task == 'global':
                return scene_gen.generate_scene_on_line(start_center=self._start_center, 
                                             nominal_goal_distance=3.5, 
                                             direction_angle_range=np.deg2rad(35), 
                                             vertical_angle_range=np.deg2rad(15), 
                                             region_radius=self._region_radius, 
                                             num_obstacles=self._num_obstacles)
        return scene_gen.generate_scene(num_obstacles=self._num_obstacles)

    # -------------------------------------------------------------- planning

    def _plan_path(self):
        (xr, yr, zr) = self.WORKSPACE
        planner = AStarPlanner(grid_size=0.2,
                               x_range=xr, y_range=yr, z_range=zr,
                               safety_margin=0.2)
        raw = planner.plan(self.START, self.GOAL, self.OBSTACLE_LIST)
        if raw is None:
            print('[GlobalPlannerAviary] A* 失败, 使用起点->终点直线作为路径')
            raw = np.stack([self.START, self.GOAL], axis=0)
        wps = AStarPlanner.resample_path(raw, spacing=self._waypoint_spacing)
        # 去掉起点 (无人机初始位置), 保留后续全部点作为追踪目标
        if len(wps) > 1:
            wps = wps[1:]
        # print(f'[GlobalPlannerAviary] 规划得到 {len(wps)} 个 waypoints:\n{wps}')
        return wps.astype(np.float32)

    # -------------------------------------------------------------- PyBullet

    def _addObstacles(self):
        """把 self.OBSTACLE_LIST 里的球体加到 PyBullet 里 (固定, 与训练/规划一致)."""
        for (center, radius) in self.OBSTACLE_LIST:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius,
                                         physicsClientId=self.CLIENT)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius,
                                      rgbaColor=[0.8, 0.1, 0.1, 0.6],
                                      physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0,  # 固定障碍
                              baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis,
                              basePosition=list(center),
                              physicsClientId=self.CLIENT)

    def _draw_planning_scene(self):
        """GUI 中绘制起点 / 终点 / waypoints / A* 路径."""
        # 起点 (绿色十字 + 文字)
        s = self.START
        p.addUserDebugLine(s + [0.15, 0, 0], s - [0.15, 0, 0],
                           lineColorRGB=[0, 1, 0], lineWidth=2,
                           physicsClientId=self.CLIENT)
        p.addUserDebugLine(s + [0, 0.15, 0], s - [0, 0.15, 0],
                           lineColorRGB=[0, 1, 0], lineWidth=2,
                           physicsClientId=self.CLIENT)
        p.addUserDebugText("START", s + [0, 0, 0.25], textColorRGB=[0, 1, 0],
                           textSize=1.2, physicsClientId=self.CLIENT)

        # 终点 (蓝色十字 + 文字)
        g = self.GOAL
        p.addUserDebugLine(g + [0.15, 0, 0], g - [0.15, 0, 0],
                           lineColorRGB=[0, 0, 1], lineWidth=2,
                           physicsClientId=self.CLIENT)
        p.addUserDebugLine(g + [0, 0.15, 0], g - [0, 0.15, 0],
                           lineColorRGB=[0, 0, 1], lineWidth=2,
                           physicsClientId=self.CLIENT)
        p.addUserDebugText("GOAL", g + [0, 0, 0.25], textColorRGB=[0, 0, 1],
                           textSize=1.5, physicsClientId=self.CLIENT)

        # 连 waypoints 的折线
        path_pts = np.vstack([self.START, self.WAYPOINTS])
        for i in range(len(path_pts) - 1):
            p.addUserDebugLine(path_pts[i], path_pts[i + 1],
                               lineColorRGB=[1, 1, 0], lineWidth=1.5,
                               physicsClientId=self.CLIENT)

        # 每个 waypoint 画小黄十字
        for i, wp in enumerate(self.WAYPOINTS):
            p.addUserDebugLine(wp + [0.08, 0, 0], wp - [0.08, 0, 0],
                               lineColorRGB=[1, 1, 0], lineWidth=1,
                               physicsClientId=self.CLIENT)
            p.addUserDebugLine(wp + [0, 0.08, 0], wp - [0, 0.08, 0],
                               lineColorRGB=[1, 1, 0], lineWidth=1,
                               physicsClientId=self.CLIENT)
            p.addUserDebugText(f"wp{i}", wp + [0, 0, 0.08],
                               textColorRGB=[1, 1, 0], textSize=0.8,
                               physicsClientId=self.CLIENT)
        
        # print('wps:\n', self.WAYPOINTS)

        # 障碍物（红色半透明球）
        for i, (center, radius) in enumerate(self.OBSTACLE_LIST):
            vis_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[1, 0, 0, 0.3],  # 半透明
                physicsClientId=self.CLIENT
            )
        
            body_id = p.createMultiBody(
                baseMass=0,  # 不参与动力学
                baseVisualShapeIndex=vis_id,
                basePosition=list(center),
                physicsClientId=self.CLIENT
            )
        
            # 可选：加标签
            p.addUserDebugText(
                f"obs{i}",
                np.array(center) + np.array([0, 0, radius + 0.1]),
                textColorRGB=[1, 0, 0],
                textSize=0.8,
                physicsClientId=self.CLIENT
            )

            

    # -------------------------------------------------------------- RL API

    def reset(self, seed=None, options=None):
        if self.IS_DEBUG:
            print('\n[GlobalPlannerAviary] reset called. 重新规划路径, 重置 waypoint 计数器.')
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_start, random_goal, random_obstacles = self._sample_scene(self.WORKSPACE)
        self.START = np.asarray(random_start, dtype=float)
        self.INIT_XYZS = self.START.reshape(1, 3)
        self.GOAL = np.asarray(random_goal, dtype=float)
        # 注意: BaseAviary 存在同名 bool 属性 self.OBSTACLES (在 super().__init__ 里赋值),
        # 这里用 OBSTACLE_LIST 避免被父类覆盖.
        self.OBSTACLE_LIST = random_obstacles if random_obstacles is not None else default_obstacles()
        # --- 跑 A*, 生成 waypoints ---
        self.WAYPOINTS = self._plan_path()

        # self.target = self.START.copy()

        self.EPISODE_LEN_SEC = 15
        self.num_waypoints = self.WAYPOINTS.shape[0]
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)
        obs, info = super().reset(seed=seed, options=options)
        if self.GUI:
            self._draw_planning_scene()

        self._reset_episode_metrics()
        return obs, info

    def _actionSpace(self):
        """覆盖父类: action = 世界坐标 PID 目标点, 范围 = workspace_bounds."""
        if self.ACT_TYPE == ActionType.RPM:
            low = np.tile(np.array([
                -1.0,-1.0,-1.0,-1.0
            ], dtype=np.float32), (self.NUM_DRONES,1))
            high = np.tile(np.array([
                1.0,1.0,1.0,1.0
            ], dtype=np.float32), (self.NUM_DRONES,1))
            # 父类在 _actionSpace 里填充 action_buffer, 这里也需要保证一致
            for _ in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, 4), dtype=np.float32))
            return spaces.Box(low=low, high=high, dtype=np.float32)
        elif self.ACT_TYPE == ActionType.PID:
            (xr, yr, zr) = self.WORKSPACE
            # low = np.tile(np.array([xr[0], yr[0], zr[0]], dtype=np.float32),
            #               (self.NUM_DRONES, 1))
            # high = np.tile(np.array([xr[1], yr[1], zr[1]], dtype=np.float32),
            #                (self.NUM_DRONES, 1))
            low = np.tile(np.array([
                -1.0,-1.0,-1.0,-0.25,-0.25,-0.25,-1.0,-1.0,-1.0
            ], dtype=np.float32), (self.NUM_DRONES,1))
            
            high = np.tile(np.array([
                1.0,1.0,1.0,0.25,0.25,0.25,1.0,1.0,1.0
            ], dtype=np.float32), (self.NUM_DRONES,1))
            # 父类在 _actionSpace 里填充 action_buffer, 这里也需要保证一致
            for _ in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES, 9), dtype=np.float32))
            return spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            return super()._actionSpace()

    def _preprocessAction(self, action):
        if self.ACT_TYPE == ActionType.RPM:
            self.action_buffer.append(action)
            rpm = np.zeros((self.NUM_DRONES, 4))
            self.model_crpy = np.zeros((self.NUM_DRONES, 4))
            self.expert_crpy = np.zeros((self.NUM_DRONES, 4))
            for k in range(self.NUM_DRONES):
                state = self._getDroneStateVector(k)
                action_asarray = np.asarray(action[k, :], dtype=float)
                # cal rpm_k
                collective_min = 11979.080
                collective_max = 21609.838
                collective_mean = (collective_min + collective_max) / 2
                collective_scale = (collective_max - collective_min)/2
                roll_scale = 253.098
                pitch_scale = 308.259
                yaw_scale = 38.606

                rpm_k = np.zeros(4)
                rpm_k[0] = collective_mean + action_asarray[0] * collective_scale - action_asarray[1] * roll_scale - action_asarray[2] * pitch_scale - action_asarray[3] * yaw_scale
                rpm_k[1] = collective_mean + action_asarray[0] * collective_scale - action_asarray[1] * roll_scale + action_asarray[2] * pitch_scale + action_asarray[3] * yaw_scale
                rpm_k[2] = collective_mean + action_asarray[0] * collective_scale + action_asarray[1] * roll_scale + action_asarray[2] * pitch_scale - action_asarray[3] * yaw_scale
                rpm_k[3] = collective_mean + action_asarray[0] * collective_scale + action_asarray[1] * roll_scale - action_asarray[2] * pitch_scale + action_asarray[3] * yaw_scale
                if self.IS_DEBUG:
                    print(f'action_asarray: {action_asarray}, rpm: {rpm_k}')

                # 直接使用PID指导，实现BC
                target_wp = self.WAYPOINTS[self.wp_counters[k]]
                expert_rpm_k, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_wp,
                )
                # 记录专家action
                
                self.expert_crpy[k, :] = np.zeros(4) 
                self.expert_crpy[k, 0] = (expert_rpm_k[0] + expert_rpm_k[1] + expert_rpm_k[2] + expert_rpm_k[3]) / 4  # collective
                self.expert_crpy[k, 1] = (-expert_rpm_k[0] - expert_rpm_k[1] + expert_rpm_k[2] + expert_rpm_k[3]) / 4  # roll
                self.expert_crpy[k, 2] = (-expert_rpm_k[0] + expert_rpm_k[1] + expert_rpm_k[2] - expert_rpm_k[3]) / 4  # pitch
                self.expert_crpy[k, 3] = (-expert_rpm_k[0] + expert_rpm_k[1] - expert_rpm_k[2] + expert_rpm_k[3]) / 4  # yaw
                self.expert_crpy[k, 0] = (self.expert_crpy[k, 0]-collective_min)/collective_scale
                self.expert_crpy[k, 1] = (self.expert_crpy[k, 1]+roll_scale)/(2*roll_scale)
                self.expert_crpy[k, 2] = (self.expert_crpy[k, 2]+pitch_scale)/(2*pitch_scale)
                self.expert_crpy[k, 3] = (self.expert_crpy[k, 3]+yaw_scale)/(2*yaw_scale)

                self.model_crpy[k, :] = (action_asarray+1.0)/2.0
                rpm[k, :] = 0.4 * rpm_k + 0.6 * expert_rpm_k  # 混合专家和模型输出, 逐渐过渡到完全使用模型输出
                self.rpm = rpm[k, :]

                if self.IS_DEBUG:
                    rpm_log_path = "test_logs/rpm_log_"+self.start_time+".csv"
                    if not os.path.exists("test_logs"):
                        os.makedirs("test_logs")
                    file_exists = os.path.exists(rpm_log_path)
                    with open(rpm_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # 第一次写表头
                        if not file_exists:
                            writer.writerow([
                                "drone_id",
                                "rpm_0", "rpm_1", "rpm_2", "rpm_3"
                            ])
                        writer.writerow([0,rpm[k, 0], rpm[k, 1], rpm[k, 2], rpm[k, 3]])
            return rpm
        elif self.ACT_TYPE == ActionType.PID:
            self.action_buffer.append(action)
            rpm = np.zeros((self.NUM_DRONES, 4))
            for k in range(self.NUM_DRONES):
                # POS_SCALE = np.array([0.5, 0.5, 0.3])
                # VEL_SCALE = np.array([0.4, 0.4, 0.2])
                # ANG_SCALE = np.array([0.2, 0.2, 0.5])
                POS_SCALE = np.array([1.0, 1.0, 1.0])
                VEL_SCALE = np.array([1.0, 1.0, 1.0])
                ANG_SCALE = np.array([1.0, 1.0, 1.0])
                state = self._getDroneStateVector(k)
                action_asarray = np.asarray(action[k, :], dtype=float)
                target_pos = action_asarray[0:3] * POS_SCALE + state[0:3]  # 目标位置 = target坐标 + action 偏移
                target_pos = self._calculateNextStep(state[0:3], target_pos, 1.0)
                # self.target = target_pos  # 更新当前目标位置
                delta_vel = action_asarray[3:6]
                next_vel = delta_vel * VEL_SCALE + state[10:13]  # 目标速度 = 当前速度 + action 指定的相对速度
                delta_ang = action_asarray[6:9]
                next_ang = delta_ang * ANG_SCALE + state[7:10]  # 目标角度 = 当前角度 + action 指定的相对角度
                if self.IS_DEBUG:
                    print(f'Drone {k} - Current Position: {state[0:3]}, Current Velocity: {state[10:13]}, Current RPY: {state[7:10]}')
                    print(f'Drone {k} - Action (pos_offset, vel_offset, ang_offset): {action_asarray}')
                    print(f'Drone {k} - Target Position: {target_pos}, Target Velocity: {next_vel}, Target RPY: {next_ang}')
                if self.IS_DEBUG:
                    log_path = "test_logs/test_log_"+self.start_time+".csv"
                    if not os.path.exists("test_logs"):
                        os.makedirs("test_logs")
                    file_exists = os.path.exists(log_path)
                
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # 第一次写表头
                        if not file_exists:
                            writer.writerow([
                                "drone_id",
                                "cur_x", "cur_y", "cur_z",
                                "vel_x", "vel_y", "vel_z",
                                "roll", "pitch", "yaw",
                                "pos_offset_x", "pos_offset_y", "pos_offset_z",
                                "vel_offset_x", "vel_offset_y", "vel_offset_z",
                                "ang_offset_roll", "ang_offset_pitch", "ang_offset_yaw",
                                "target_x", "target_y", "target_z",
                                "vel_target_x", "vel_target_y", "vel_target_z",
                                "ang_target_roll", "ang_target_pitch", "ang_target_yaw"
                            ])
                        writer.writerow([
                            k,
                    
                            state[0],
                            state[1],
                            state[2],
                    
                            state[10],
                            state[11],
                            state[12],
                    
                            state[7],
                            state[8],
                            state[9],
        
                            action_asarray[0],
                            action_asarray[1],
                            action_asarray[2],
        
                            action_asarray[3],
                            action_asarray[4],
                            action_asarray[5],
        
                            action_asarray[6],
                            action_asarray[7],
                            action_asarray[8],
                    
                            target_pos[0],
                            target_pos[1],
                            target_pos[2],
                        
                            next_vel[0],
                            next_vel[1],
                            next_vel[2],
        
                            next_ang[0],
                            next_ang[1],
                            next_ang[2]
                        ])
                rpm_k, _, _ = self.ctrl[k].computeControl(
                    control_timestep=self.CTRL_TIMESTEP,
                    cur_pos=state[0:3],
                    cur_quat=state[3:7],
                    cur_vel=state[10:13],
                    cur_ang_vel=state[13:16],
                    target_pos=target_pos,
                    target_rpy=next_ang,
                    target_vel=next_vel
                )
                rpm[k, :] = rpm_k
                self.rpm = rpm[k, :]
                if self.IS_DEBUG:
                    rpm_log_path = "test_logs/rpm_log_"+self.start_time+".csv"
                    if not os.path.exists("test_logs"):
                        os.makedirs("test_logs")
                    file_exists = os.path.exists(rpm_log_path)
                    with open(rpm_log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        # 第一次写表头
                        if not file_exists:
                            writer.writerow([
                                "drone_id",
                                "rpm_0", "rpm_1", "rpm_2", "rpm_3"
                            ])
                        writer.writerow([0,rpm[k, 0], rpm[k, 1], rpm[k, 2], rpm[k, 3]])
            return rpm
        else:
            return super()._preprocessAction(action)

    def step(self, action):
        obs, reward, terminated, truncated, info = \
            super().step(action)
    
        self._update_episode_metrics()
    
        done = terminated or truncated
    
        if done:
            self._finalize_episode_metrics()
            if self.mode == "testing":
                self._save_episode_metrics()
    
        return obs, reward, terminated, truncated, info


    # ---- observation --------------------------------------------------------

    def _observationSpace(self):
        obs_space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            shape = obs_space.shape
            new_dim = shape[-1] + 3  # 追加当前目标的相对位置
            # print(f'[GlobalPlannerAviary] 观察空间维度从 {shape[-1]} 扩展到 {new_dim} (追加目标相对位置 + 速度/角度偏移)')
            return spaces.Box(low=-np.inf, high=np.inf,
                              shape=(self.NUM_DRONES, new_dim), dtype=np.float32)
        return obs_space

    def _computeObs(self):
        root_obs = super()._computeObs()
        target_info = np.zeros((self.NUM_DRONES, 3), dtype=np.float32)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            cur_wp = self.WAYPOINTS[self.wp_counters[i]]
            target_info[i, :] = cur_wp - state[0:3]
        if self.IS_DEBUG:
            print(f'for drone{i}, final_obs: {np.hstack([root_obs, target_info]).astype(np.float32)}')
        return np.hstack([root_obs, target_info]).astype(np.float32)

    # ---- reward -------------------------------------------------------------

    def _min_obs_distance(self, pos):
        d_min = np.inf
        
        # --- 1. 计算原有圆形障碍物的距离 ---
        for (c, r) in self.OBSTACLE_LIST:
            d = np.linalg.norm(pos - np.asarray(c)) - r
            if d < d_min:
                d_min = d
                
        # --- 2. 计算到边界（WORKSPACE）的距离 ---
        # self.WORKSPACE 结构: ((-1.0, 5.0), (-3.0, 3.0), (0.1, 2.5))
        (xr, yr, zr) = self.WORKSPACE
        
        # 到 6 个面的距离：坐标值减去最小值，或最大值减去坐标值
        dist_to_bounds = [
            pos[0] - xr[0], # 到左墙 (x_min)
            xr[1] - pos[0], # 到右墙 (x_max)
            pos[1] - yr[0], # 到前墙 (y_min)
            yr[1] - pos[1], # 到后墙 (y_max)
            pos[2] - zr[0], # 到地板 (z_min)
            zr[1] - pos[2]  # 到天花板 (z_max)
        ]
        
        min_bound_dist = min(dist_to_bounds)
        
        # 取障碍物距离和边界距离的最小值
        if min_bound_dist < d_min:
            d_min = min_bound_dist
            
        return d_min

    def _computeReward(self):
        """量纲调平后的 reward.

        设计思路:
            - 距离 shaping 用 (dist_prev - dist_now) * k_dist, k_dist 适中, 让单 step 量级 ~ 0.01
            - 终点 / waypoint 奖励远大于 shaping 累计, 强迫 agent 真正到达
            - 撞障碍给一次性大额惩罚 (episode 也会被 terminate)
            - 姿态 / 高度只做温和正则, 避免与距离 shaping 冲突
        """
        rewards = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[10:13]
            target_wp = self.WAYPOINTS[self.wp_counters[i]]
            to_wp = target_wp - pos
            dist = np.linalg.norm(to_wp)
            speed = float(np.linalg.norm(vel))
            vel_dir = vel / (speed + 1e-8) # 最好别用这个
            dir_wp = to_wp / (dist + 1e-8)
            learn_type = ""
            if learn_type!="teaching":
                # 0) 距离奖励
                next_pos = pos + vel * self.CTRL_TIMESTEP
                next_dist = np.linalg.norm(target_wp - next_pos)
                rewards[i] += -1 * (next_dist - dist)*2500
                if self.IS_DEBUG:
                    print('当前位置：', pos, '目标位置：', target_wp, '距离：', dist, '下一步距离：', next_dist)
                    print('距离奖励：', -1* (next_dist - dist)*2500)
    
                # 1) 朝向 waypoint 的速度投影奖励
                rewards[i] += float(np.dot(vel, dir_wp)) *50.0
                if self.IS_DEBUG:
                    print('朝向奖励：', float(np.dot(vel, dir_wp)) *50.0)
    
                # 2) 高度
                # rewards[i] += -(pos[2] - target_wp[2])**2 * 5.0
                # if self.IS_DEBUG:
                #     print('高度奖励：', -(pos[2] - target_wp[2])**2 * 5.0)
    
                # 3) 到达 waypoint 奖励
                if dist < self.ARRIVAL_RADIUS:
                    if self.wp_counters[i] < self.num_waypoints - 1:
                        rewards[i] += 100.0
                        self.wp_counters[i] += 1
                        if self.IS_DEBUG:
                            print('到达 waypoint 奖励：100.0, 下一个目标：wp%d' % self.wp_counters[i], 'wp位置：', target_wp)
                    else:
                        rewards[i] += 1000.0
                        if self.IS_DEBUG:
                            print('到达最终目标奖励：1000.0')
    
                # 4) 避障惩罚: 离最近障碍越近罚越多, 进入障碍体则大额惩罚
                obs_threshold = 0.2
                d_obs = self._min_obs_distance(pos)
                if d_obs < 0.0:
                    rewards[i] -= self.COLLISION_PENALTY
                elif d_obs < obs_threshold:
                    rewards[i] -= (obs_threshold - d_obs) * 1.5
                if self.IS_DEBUG:
                    print('避障惩罚：', -self.COLLISION_PENALTY if d_obs < 0.0 else (- (obs_threshold - d_obs) * 1.5 if d_obs < obs_threshold else 0.0))
    
                # 5) 坠地 / 飞出边界的软惩罚
                (xr, yr, zr) = self.WORKSPACE
                if (pos[0] < xr[0] or pos[0] > xr[1]
                    or pos[1] < yr[0] or pos[1] > yr[1]
                    or pos[2] < zr[0] or pos[2] > zr[1]):
                    rewards[i] -= 5.0
                if self.IS_DEBUG:
                    print('边界惩罚：', -5.0 if (pos[0] < xr[0] or pos[0] > xr[1]
                        or pos[1] < yr[0] or pos[1] > yr[1]
                        or pos[2] < zr[0] or pos[2] > zr[1]) else 0.0)
                    
                # 5.1) 反转惩罚
                q = state[3:7]
                if 1 - 2*(q[0]**2 + q[1]**2) < 0.0:  # roll 超过 90 度
                    rewards[i] -= 500.0
                if self.IS_DEBUG:
                    print('反转惩罚：', -500.0 if 1 - 2*(q[0]**2 + q[1]**2) < 0.0 else 0.0)
    
                # 5.2) 过慢惩罚
                # if speed < 0.25:
                #     rewards[i] -= (0.25 - speed)**2 *50.0
                # if self.IS_DEBUG:
                #     print('过慢惩罚：', -(0.25 - speed)**2 *50.0 if speed < 0.5 else 0.0)
    
                # 6) 角度惩罚
                row, pitch, yaw = state[7:10]
                rewards[i]+=-1*(1.0 * row**2 + 1.0 * pitch**2 + 0.1 * yaw**2) * 300.0
                if self.IS_DEBUG:
                    print(f'角度惩罚：{-1*(1.0 * row**2 + 1.0 * pitch**2 + 0.1 * yaw**2) * 300.0}')
                
                # 7) 生存惩罚
                rewards[i] -= 0.2
                if self.IS_DEBUG:
                    print('生存惩罚：', -0.2)
    
                if self.IS_DEBUG:
                    print(f"drone{i} dist={dist:.2f} d_obs={d_obs:.2f} "
                          f"wp={self.wp_counters[i]} wp_pos={target_wp} r={rewards[i]:.2f}")
            else:
                expert_norm = self.expert_crpy[i, :]
                model_norm = self.model_crpy[i, :]
                if self.IS_DEBUG:
                    print(f'expert_norm: {expert_norm}, model_norm: {model_norm}')
                expert_norm[0]*=3.0
                model_norm[0]*=3.0
                rewards[i] += -100.0*np.mean((model_norm-expert_norm)**2)
                if self.IS_DEBUG:
                    print(f'行为克隆奖励：{-100.0*np.mean((model_norm-expert_norm)**2)}')
                if dist < self.ARRIVAL_RADIUS:
                    if self.wp_counters[i] < self.num_waypoints - 1:
                        self.wp_counters[i] += 1
                        if self.IS_DEBUG:
                            print('到达 waypoint , 下一个目标：wp%d' % self.wp_counters[i], 'wp位置：', target_wp)
                    else:
                        if self.IS_DEBUG:
                            print('到达最终目标')
        return float(rewards.sum()) if self.NUM_DRONES == 1 else rewards

    # ---- termination --------------------------------------------------------

    def _computeTerminated(self):
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            # 撞障碍 -> 终止
            if self._min_obs_distance(pos) < 0.0:
                return True
            # 坠地
            (xr, yr, zr) = self.WORKSPACE
            if (pos[0] < xr[0] or pos[0] > xr[1]
                or pos[1] < yr[0] or pos[1] > yr[1]
                or pos[2] < zr[0] or pos[2] > zr[1]):
                return True
            # 反转
            q = state[3:7]
            if 1 - 2*(q[0]**2 + q[1]**2) < 0.0:  # roll 超过 90 度
                return True
        # 到达最终目标
        done_all = True
        for i in range(self.NUM_DRONES):
            pos = self._getDroneStateVector(i)[0:3]
            reached_last = (self.wp_counters[i] == self.num_waypoints - 1
                            and np.linalg.norm(self.WAYPOINTS[-1] - pos)
                            < self.ARRIVAL_RADIUS)
            done_all = done_all and reached_last
        return done_all

    def _computeTruncated(self):
        truncated = (
            self.step_counter / self.PYB_FREQ
            > self.EPISODE_LEN_SEC
        )
        if truncated:
            self.episode_metrics["termination_reason"] = "timeout"
        return truncated

    def _computeInfo(self):
        return {
            "current_wp_indices": self.wp_counters.copy(),
            "num_waypoints": self.num_waypoints,
            "episode_metrics": self.episode_metrics.copy()
        }
