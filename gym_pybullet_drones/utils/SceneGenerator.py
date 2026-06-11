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
        print('using this!!!')
        """
        随机生成一套完整的场景配置
        """
        # 1. 生成起点 (固定 z=0)
        start = np.array([
            random.uniform(self.x_range[0] + 0.5, self.x_range[0] + 1.5),
            random.uniform(self.y_range[0] + 0.5, self.y_range[1] - 0.5),
            0.0135
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
    
    def generate_takeoff(
        self,
        start_center=(2.0, 0.0),
        hover_height_range=(0.5, 1.5)):
        """
        起飞 -> 悬停
        """
    
        start = np.array([
            start_center[0] + np.random.uniform(-0.5, 0.5),
            start_center[1] + np.random.uniform(-0.5, 0.5),
            0.0135
        ])
    
        goal = np.array([
            start[0] + np.random.uniform(-2.0,2.0),
            start[1] + np.random.uniform(-2.0, 2.0),
            np.random.uniform(*hover_height_range)
        ])
    
        obstacles = []
    
        return start, goal, obstacles

    def generate_short_distance(
        self,
        start_center=(2.0, 0.0),
        distance_range=(1.5, 3.5)):
        """
        短距离导航
        """
    
        start = np.array([
            start_center[0] + np.random.uniform(-0.2, 0.2),
            start_center[1] + np.random.uniform(-0.2, 0.2),
            0.0135
        ])
    
        yaw = np.random.uniform(-np.pi, np.pi)
    
        dist = np.random.uniform(*distance_range)
    
        goal = start + np.array([
            dist * np.cos(yaw),
            dist * np.sin(yaw),
            np.random.uniform(0.5, 1.0)
        ])
    
        goal[0] = np.clip(goal[0], -0.5, 4.5)
        goal[1] = np.clip(goal[1], -2.5, 2.5)
        goal[2] = np.clip(goal[2], 0.5, 2.0)
    
        return start, goal, []

    def generate_scene_on_line(
            self,
            start_center=(0.0, 0.0, 1.0),
            nominal_goal_distance=3.5,
            direction_angle_range=np.deg2rad(35),   # 最大偏转角
            vertical_angle_range=np.deg2rad(15),    # z方向轻微扰动
            region_radius=0.3,
            num_obstacles=2,
            obs_radius_range=(0.25, 0.4),
            t_range=(0.25, 0.75),
            perp_offset_max=0.05,
            clearance=0.15):
    
        """
        改进版:
            - start 在小区域随机
            - goal 不再固定在 +x
            - 而是在“主方向附近”随机偏转
            - 难度温和增加
        """
    
        # -------------------------------------------------
        # 1. start 随机
        # -------------------------------------------------
    
        offset_xy = np.random.uniform(
            -region_radius,
            region_radius,
            size=2
        )
    
        start = np.array([
            start_center[0] + offset_xy[0],
            start_center[1] + offset_xy[1],
            0.0135
        ])
    
        # -------------------------------------------------
        # 2. goal 方向轻微随机化
        # -------------------------------------------------
    
        # 水平偏转角
        yaw = np.random.uniform(
            -direction_angle_range,
            direction_angle_range
        )
    
        # 垂直偏转角
        pitch = np.random.uniform(
            -vertical_angle_range,
            vertical_angle_range
        )
    
        # 距离也稍微随机
        dist = np.random.uniform(
            nominal_goal_distance - 0.5,
            nominal_goal_distance + 0.5
        )
    
        # 球坐标 -> 笛卡尔
        direction = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])
    
        goal = start + direction * dist
    
        # -------------------------------------------------
        # 3. goal 边界裁剪
        # -------------------------------------------------
    
        goal[0] = np.clip(goal[0],
                          self.x_range[0] + 0.5,
                          self.x_range[1] - 0.5)
    
        goal[1] = np.clip(goal[1],
                          self.y_range[0] + 0.5,
                          self.y_range[1] - 0.5)
    
        goal[2] = np.clip(goal[2],
                          self.z_range[0] + 0.5,
                          self.z_range[1] - 0.5)
    
        # -------------------------------------------------
        # 4. 连线方向
        # -------------------------------------------------
    
        line_vec = goal - start
        line_len = np.linalg.norm(line_vec)
    
        if line_len < 1e-6:
            return start, goal, []
    
        line_dir = line_vec / line_len
    
        # -------------------------------------------------
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
    
    def generate_fix(self):
        """
        生成一个固定路径（确定性），并将工作空间设置为
        workspace = ((-1.0, 5.0), (-3.0, 3.0), (-0.1, 2.5))

        返回: start, goal, obstacles
        """
        # 固定 workspace
        self.x_range = (-1.0, 5.0)
        self.y_range = (-3.0, 3.0)
        self.z_range = (-0.1, 2.5)

        # 固定起点（靠近左侧平面，低高度）
        start = np.array([self.x_range[0] + 0.5, 0.0, 0.0135])

        # 固定终点（靠近右侧平面，中高度）
        goal = np.array([self.x_range[1] - 0.5, 0.0, 1.2])
        # 固定一些障碍物，位置在工作空间内
        obstacles = []

        obstacles.append((np.array([3.5, 0.0, 1.0]), 0.35))
        obstacles.append((np.array([1.8, 0.3, 0.3]), 0.3))
        return start, goal, obstacles
    
    def generate_fix_randomized(self):
    
        start = np.array([
            -0.5 + np.random.uniform(-0.0,0.25),
            np.random.uniform(-0.25,0.25),
            0.0135
        ])
    
        goal = np.array([
            4.5 + np.random.uniform(-0.25,0.0),
            np.random.uniform(-0.5,0.5),
            np.random.uniform(0.6,1.0)
        ])
    
        obstacles = []
    
        obstacles.append((
            np.array([
                1.8 + np.random.uniform(-0.25,0.25),
                0.3 + np.random.uniform(-0.25,0.25),
                0.3 + np.random.uniform(-0.25,0.25)
            ]),
            np.random.uniform(0.25,0.35)
        ))
    
        obstacles.append((
            np.array([
                3.5 + np.random.uniform(-0.25,0.25),
                0.0 + np.random.uniform(-0.25,0.25),
                1.0 + np.random.uniform(-0.25,0.25)
            ]),
            np.random.uniform(0.30,0.40)
        ))

        print('start = ', start)
        print('goal = ', goal)
        print('obstacles = ', obstacles)
    
        return start, goal, obstacles