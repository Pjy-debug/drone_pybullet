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

from envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.AStarPlanner import AStarPlanner


IS_DEBUG = False


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
                 arrival_radius: float = 0.25,
                 workspace_bounds=((-1.0, 5.0), (-3.0, 3.0), (0.1, 2.5)),
                 act_scale: float = None,   # 预留, 当前未使用
                 collision_penalty: float = 50.0,
                 **kwargs):
        # --- 规划相关参数必须在父类调用前准备好 (父类会调 _addObstacles / _actionSpace) ---
        self.START = np.asarray(start, dtype=float)
        self.GOAL = np.asarray(goal, dtype=float)
        self.OBSTACLES = obstacles if obstacles is not None else default_obstacles()
        self.ARRIVAL_RADIUS = arrival_radius
        self.COLLISION_PENALTY = collision_penalty
        self.WORKSPACE = workspace_bounds
        self._waypoint_spacing = waypoint_spacing

        # --- 跑 A*, 生成 waypoints ---
        self.WAYPOINTS = self._plan_path()

        # 让无人机从 start 起飞
        kwargs.setdefault('initial_xyzs', self.START.reshape(1, 3))

        super().__init__(**kwargs)

        self.EPISODE_LEN_SEC = 25
        self.num_waypoints = self.WAYPOINTS.shape[0]
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)

        # 可视化 id 记录
        self._dbg_item_ids = []
        if self.GUI:
            self._draw_planning_scene()

    # -------------------------------------------------------------- planning

    def _plan_path(self):
        (xr, yr, zr) = self.WORKSPACE
        planner = AStarPlanner(grid_size=0.2,
                               x_range=xr, y_range=yr, z_range=zr,
                               safety_margin=0.2)
        raw = planner.plan(self.START, self.GOAL, self.OBSTACLES)
        if raw is None:
            print('[GlobalPlannerAviary] A* 失败, 使用起点->终点直线作为路径')
            raw = np.stack([self.START, self.GOAL], axis=0)
        wps = AStarPlanner.resample_path(raw, spacing=self._waypoint_spacing)
        # 去掉起点 (无人机初始位置), 保留后续全部点作为追踪目标
        if len(wps) > 1:
            wps = wps[1:]
        print(f'[GlobalPlannerAviary] 规划得到 {len(wps)} 个 waypoints:\n{wps}')
        return wps.astype(np.float32)

    # -------------------------------------------------------------- PyBullet

    def _addObstacles(self):
        """把 self.OBSTACLES 里的球体加到 PyBullet 里 (固定, 与训练/规划一致)."""
        for (center, radius) in self.OBSTACLES:
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

    # -------------------------------------------------------------- RL API

    def reset(self, seed=None, options=None):
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)
        obs, info = super().reset(seed=seed, options=options)
        if self.GUI:
            self._draw_planning_scene()
        return obs, info

    # ---- action: 不再裁剪到 [-1,1], 直接作为世界坐标 PID 目标 ---------------

    def _actionSpace(self):
        """覆盖父类: action = 世界坐标 PID 目标点, 范围 = workspace_bounds."""
        # 仅支持 PID 模式
        if self.ACT_TYPE != ActionType.PID:
            return super()._actionSpace()
        (xr, yr, zr) = self.WORKSPACE
        low = np.tile(np.array([xr[0], yr[0], zr[0]], dtype=np.float32),
                      (self.NUM_DRONES, 1))
        high = np.tile(np.array([xr[1], yr[1], zr[1]], dtype=np.float32),
                       (self.NUM_DRONES, 1))
        # 父类在 _actionSpace 里填充 action_buffer, 这里也需要保证一致
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES, 3), dtype=np.float32))
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _preprocessAction(self, action):
        """PID 模式下: action 直接作为目标位置. 其他模式回退到父类."""
        if self.ACT_TYPE != ActionType.PID:
            return super()._preprocessAction(action)

        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            state = self._getDroneStateVector(k)
            target_pos = np.asarray(action[k, :], dtype=float)
            next_pos = self._calculateNextStep(current_position=state[0:3],
                                               destination=target_pos,
                                               step_size=1)
            rpm_k, _, _ = self.ctrl[k].computeControl(
                control_timestep=self.CTRL_TIMESTEP,
                cur_pos=state[0:3],
                cur_quat=state[3:7],
                cur_vel=state[10:13],
                cur_ang_vel=state[13:16],
                target_pos=next_pos,
            )
            rpm[k, :] = rpm_k
        return rpm

    # ---- observation --------------------------------------------------------

    def _observationSpace(self):
        obs_space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            shape = obs_space.shape
            new_dim = shape[-1] + 3  # 追加当前目标的相对位置
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
        return np.hstack([root_obs, target_info]).astype(np.float32)

    # ---- reward -------------------------------------------------------------

    def _min_obs_distance(self, pos):
        d_min = np.inf
        for (c, r) in self.OBSTACLES:
            d = np.linalg.norm(pos - np.asarray(c)) - r
            if d < d_min:
                d_min = d
        return d_min

    def _computeReward(self):
        rewards = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            vel = state[10:13]
            target_wp = self.WAYPOINTS[self.wp_counters[i]]
            to_wp = target_wp - pos
            dist = np.linalg.norm(to_wp)
            dir_wp = to_wp / (dist + 1e-6)
            speed = np.linalg.norm(vel)
            vel_dir = vel / (speed + 1e-6)

            # 1) 朝向 waypoint 的速度投影奖励
            rewards[i] += float(np.dot(vel_dir, dir_wp) > 0) * speed

            # 2) 距离塑形 (越近越好, 小尺度)
            rewards[i] += -0.1 * dist

            # 3) 到达 waypoint 奖励
            if dist < self.ARRIVAL_RADIUS:
                if self.wp_counters[i] < self.num_waypoints - 1:
                    rewards[i] += 10.0
                    self.wp_counters[i] += 1
                else:
                    rewards[i] += 100.0

            # 4) 避障惩罚: 离最近障碍越近罚越多, 进入障碍体则大额惩罚
            d_obs = self._min_obs_distance(pos)
            if d_obs < 0.0:
                rewards[i] -= self.COLLISION_PENALTY
            elif d_obs < 0.3:
                rewards[i] -= (0.3 - d_obs) * 5.0

            # 5) 坠地 / 飞出边界的软惩罚
            (xr, yr, zr) = self.WORKSPACE
            if (pos[0] < xr[0] or pos[0] > xr[1]
                or pos[1] < yr[0] or pos[1] > yr[1]
                or pos[2] < zr[0] or pos[2] > zr[1]):
                rewards[i] -= 5.0

            if IS_DEBUG:
                print(f"drone{i} dist={dist:.2f} d_obs={d_obs:.2f} "
                      f"wp={self.wp_counters[i]} r={rewards[i]:.2f}")
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
            if pos[2] < 0.05:
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
        return self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC

    def _computeInfo(self):
        return {
            "current_wp_indices": self.wp_counters.copy(),
            "num_waypoints": self.num_waypoints,
        }
