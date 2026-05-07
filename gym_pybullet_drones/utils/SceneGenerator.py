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