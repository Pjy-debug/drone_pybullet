import numpy as np
import heapq
import pybullet as p

class AStarPlanner:
    def __init__(self, grid_size=0.2, x_range=(-5, 5), y_range=(-5, 5), z_range=(0, 3)):
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        # 计算网格维度
        self.nx = int((x_range - x_range) / grid_size)
        self.ny = int((y_range - y_range) / grid_size)
        self.nz = int((z_range - z_range) / grid_size)

    def _pos_to_index(self, pos):
        ix = int((pos - self.x_range) / self.grid_size)
        iy = int((pos - self.y_range) / self.grid_size) # 修正偏移
        iz = int((pos - self.z_range) / self.grid_size)
        return (ix, iy, iz)

    def _index_to_pos(self, index):
        x = index * self.grid_size + self.x_range
        y = index * self.grid_size + self.y_range
        z = index * self.grid_size + self.z_range
        return np.array([x, y, z])

    def is_collision(self, index, obstacles):
        """
        简单的碰撞检测：如果在网格范围内有障碍物物体，则判定为碰撞
        在实际 PyBullet 中，建议使用 p.getClosestPoints 或预扫描地图
        """
        pos = self._index_to_pos(index)
        # 这里假设 obstacles 是一个包含障碍物位置和半径的列表
        for (obs_pos, radius) in obstacles:
            if np.linalg.norm(pos - obs_pos) < radius + 0.1: # 0.1 为无人机安全余量
                return True
        return False

    def plan(self, start_pos, goal_pos, obstacles):
        start_idx = self._pos_to_index(start_pos)
        goal_idx = self._pos_to_index(goal_pos)
        
        open_list = []
        heapq.heappush(open_list, (0, start_idx))
        came_from = {start_idx: None}
        g_score = {start_idx: 0}
        
        # 26个邻居方向（3D 空间）
        directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0: continue
                    directions.append((dx, dy, dz))

        while open_list:
            current = heapq.heappop(open_list)

            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(self._index_to_pos(current))
                    current = came_from[current]
                return np.array(path[::-1]) # 返回从起点到终点的路径

            for dx, dy, dz in directions:
                neighbor = (current + dx, current + dy, current + dz)
                
                # 边界检查
                if 0 <= neighbor < self.nx and 0 <= neighbor < self.ny and 0 <= neighbor < self.nz:
                    if self.is_collision(neighbor, obstacles):
                        continue
                        
                    tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal_idx))
                        came_from[neighbor] = current
                        heapq.heappush(open_list, (f_score, neighbor))
        print('warning: no path found')            
        return None # 未找到路径