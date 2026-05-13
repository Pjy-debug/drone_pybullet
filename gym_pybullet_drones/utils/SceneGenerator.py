import numpy as np
import random

class SceneGenerator:
    def __init__(self, 
                 x_range=(0.0, 5.0), 
                 y_range=(-2.0, 2.0), 
                 z_range=(0.0, 2.0), 
                 min_dist=0.8):
        """
        Args:
            x_range, y_range, z_range: 工作空间边界 (min, max)
            min_dist: 起点、终点与障碍物之间的最小安全距离
        """
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.min_dist = min_dist

    def generate_bounds(self):
        """返回符合环境要求的 workspace_bounds 格式"""
        return (self.x_range, self.y_range, self.z_range)

    def generate_scene(self, num_obstacles=3):
        """
        随机生成一套完整的场景配置
        """
        # 1. 生成起点 (固定 z=0)
        start = np.array([
            random.uniform(self.x_range[0] + 0.5, self.x_range[0] + 1.5),
            random.uniform(self.y_range[0] + 0.5, self.y_range[1] - 0.5),
            0.0
        ])

        # 2. 生成终点 (确保距离起点足够远)
        while True:
            goal = np.array([
                random.uniform(self.x_range[1] - 1.5, self.x_range[1] - 0.5),
                random.uniform(self.y_range[0] + 0.5, self.y_range[1] - 0.5),
                random.uniform(self.z_range[0] + 0.5, self.z_range[1] - 0.5)
            ])
            if np.linalg.norm(goal - start) > 2.0:
                break

        # 3. 生成障碍物 (不与起点、终点重叠)
        obstacles = []
        attempts = 0
        while len(obstacles) < num_obstacles and attempts < 100:
            attempts += 1
            # 随机障碍物中心
            obs_pos = np.array([
                random.uniform(self.x_range[0] + 0.5, self.x_range[1] - 0.5),
                random.uniform(self.y_range[0] + 0.5, self.y_range[1] - 0.5),
                random.uniform(self.z_range[0] + 0.2, self.z_range[1] - 0.2)
            ])
            obs_radius = random.uniform(0.25, 0.45)

            # 检查与起点的距离
            if np.linalg.norm(obs_pos - start) < (obs_radius + self.min_dist):
                continue
            # 检查与终点的距离
            if np.linalg.norm(obs_pos - goal) < (obs_radius + self.min_dist):
                continue
            
            obstacles.append((obs_pos, obs_radius))

        return start, goal, obstacles

    @staticmethod
    def _sample_in_ball(center, radius):
        """在以 center 为圆心、radius 为半径的球内均匀采样一点."""
        center = np.asarray(center, dtype=float)
        # 拒绝采样, 简单可靠
        while True:
            offset = np.random.uniform(-radius, radius, size=3)
            if np.linalg.norm(offset) <= radius:
                return center + offset

    def generate_scene_on_line(self,
                               start_center=(0.0, 0.0, 1.0),
                               goal_center=(3.5, 0.0, 1.0),
                               region_radius=0.3,
                               num_obstacles=2,
                               obs_radius_range=(0.25, 0.4),
                               t_range=(0.25, 0.75),
                               perp_offset_max=0.15,
                               clearance=0.15):
        """简化任务: start / goal 在两个小球形区域内变化, 障碍物生成在 start-goal 连线附近.

        Args:
            start_center, goal_center: 起点 / 终点的中心位置.
            region_radius: 起点 / 终点扰动的球半径 (越小任务越简单).
            num_obstacles: 障碍数量.
            obs_radius_range: 障碍球半径采样区间.
            t_range: 障碍中心沿连线的参数化位置范围 t∈[0,1] (0=start, 1=goal).
                避开两端, 默认 [0.25, 0.75].
            perp_offset_max: 障碍中心在垂直于连线方向上的最大扰动 (越小越容易挡住路径).
            clearance: 障碍表面与 start / goal 的最小间隙.
        """
        offset_xy = np.random.uniform(
            -region_radius,
            region_radius,
            size=2
        )
        start = np.array([
            start_center[0] + offset_xy[0],
            start_center[1] + offset_xy[1],
            0.0
        ])
        goal = self._sample_in_ball(goal_center, region_radius)

        line_vec = goal - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            # 极端兜底: start ≈ goal, 直接退化为不带障碍的场景
            return start, goal, []
        line_dir = line_vec / line_len

        # 构造一组与 line_dir 正交的基 (e1, e2)
        helper = np.array([0.0, 0.0, 1.0]) if abs(line_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(line_dir, helper)
        e1 /= (np.linalg.norm(e1) + 1e-9)
        e2 = np.cross(line_dir, e1)
        e2 /= (np.linalg.norm(e2) + 1e-9)

        obstacles = []
        attempts = 0
        max_attempts = 200
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            t = np.random.uniform(t_range[0], t_range[1])
            obs_radius = np.random.uniform(*obs_radius_range)

            # 垂直平面内的小扰动
            a = np.random.uniform(-perp_offset_max, perp_offset_max)
            b = np.random.uniform(-perp_offset_max, perp_offset_max)
            obs_pos = start + t * line_vec + a * e1 + b * e2

            # 工作空间边界检查 (留一点裕度)
            if not (self.x_range[0] + obs_radius <= obs_pos[0] <= self.x_range[1] - obs_radius
                    and self.y_range[0] + obs_radius <= obs_pos[1] <= self.y_range[1] - obs_radius
                    and self.z_range[0] + obs_radius <= obs_pos[2] <= self.z_range[1] - obs_radius):
                continue

            # 不能盖住 start / goal
            if np.linalg.norm(obs_pos - start) < obs_radius + clearance:
                continue
            if np.linalg.norm(obs_pos - goal) < obs_radius + clearance:
                continue

            # 障碍之间也不要重叠太多 (允许稍微靠近)
            ok = True
            for (c, r) in obstacles:
                if np.linalg.norm(obs_pos - c) < (obs_radius + r) * 0.8:
                    ok = False
                    break
            if not ok:
                continue

            obstacles.append((obs_pos, obs_radius))

        return start, goal, obstacles