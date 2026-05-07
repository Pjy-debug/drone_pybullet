import heapq
import numpy as np

class AStarPlanner:
    def __init__(self,
                 grid_size: float = 0.2,
                 x_range=(-1.0, 5.0),
                 y_range=(-3.0, 3.0),
                 z_range=(0.0, 2.5),
                 safety_margin: float = 0.2,
                 min_height: float = 0.25,
                 low_alt_penalty: float = 10.0):
        """
        3D 栅格 A* 路径规划器 (支持地面起飞与低空惩罚)

        Parameters
        ----------
        grid_size : float
            栅格分辨率 (米)。
        x_range, y_range, z_range : tuple
            规划区域边界。注意 z_range[0] 应包含地面高度。
        safety_margin : float
            障碍物膨胀半径。
        min_height : float
            理想的最小飞行高度（巡航高度下限）。
        low_alt_penalty : float
            低空飞行的额外代价权重。值越大，无人机越倾向于尽快爬升。
        """
        self.grid_size = grid_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.safety_margin = safety_margin
        self.min_height = min_height
        self.penalty = low_alt_penalty

        # 计算网格数量
        self.nx = max(1, int(np.ceil((self.x_max - self.x_min) / grid_size)))
        self.ny = max(1, int(np.ceil((self.y_max - self.y_min) / grid_size)))
        self.nz = max(1, int(np.ceil((self.z_max - self.z_min) / grid_size)))

    # ------------------------------------------------------------------ 核心工具

    def _pos_to_index(self, pos):
        ix = int(round((pos[0] - self.x_min) / self.grid_size))
        iy = int(round((pos[1] - self.y_min) / self.grid_size))
        iz = int(round((pos[2] - self.z_min) / self.grid_size))
        return (int(np.clip(ix, 0, self.nx - 1)),
                int(np.clip(iy, 0, self.ny - 1)),
                int(np.clip(iz, 0, self.nz - 1)))

    def _index_to_pos(self, idx):
        return np.array([self.x_min + idx[0] * self.grid_size,
                         self.y_min + idx[1] * self.grid_size,
                         self.z_min + idx[2] * self.grid_size])

    def _in_bounds(self, idx):
        return (0 <= idx[0] < self.nx and 
                0 <= idx[1] < self.ny and 
                0 <= idx[2] < self.nz)

    def _is_collision(self, idx, obstacles):
        """检测当前栅格是否与球形障碍物相交"""
        pos = self._index_to_pos(idx)
        for (obs_pos, radius) in obstacles:
            if np.linalg.norm(pos - np.asarray(obs_pos)) < radius + self.safety_margin:
                return True
        return False

    # ------------------------------------------------------------------ 规划主程序

    def plan(self, start_pos, goal_pos, obstacles=None):
        """
        运行 A* 算法。
        
        Returns: 
            shape=(N,3) 的路径点数组，如果失败则返回 None。
        """
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        obstacles = obstacles if obstacles is not None else []

        start_idx = self._pos_to_index(start_pos)
        goal_idx = self._pos_to_index(goal_pos)

        # 26 邻居寻找方向
        directions = [(dx, dy, dz) 
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) 
                      if not (dx == 0 and dy == 0 and dz == 0)]

        # 启发式函数 (Euclidean Distance)
        def h(idx):
            # 将索引差转换为物理距离
            diff = (np.array(idx) - np.array(goal_idx)) * self.grid_size
            return np.linalg.norm(diff)

        # 优先队列 (f_score, g_score, current_idx)
        open_heap = []
        heapq.heappush(open_heap, (h(start_idx), 0.0, start_idx))
        
        came_from = {start_idx: None}
        g_score = {start_idx: 0.0}

        while open_heap:
            f, cur_g, current = heapq.heappop(open_heap)

            if current == goal_idx:
                # 找到路径，开始回溯
                path_idx = []
                node = current
                while node is not None:
                    path_idx.append(node)
                    node = came_from[node]
                path_idx.reverse()
                
                path = np.array([self._index_to_pos(i) for i in path_idx])
                # 修正首尾精确位置
                path[0] = start_pos
                path[-1] = goal_pos
                return path

            for d in directions:
                nb = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
                
                if not self._in_bounds(nb) or self._is_collision(nb, obstacles):
                    continue

                # 计算基础步长 (考虑对角线移动)
                step_dist = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2) * self.grid_size
                
                # --- 高度代价逻辑 ---
                nb_pos = self._index_to_pos(nb)
                cost_multiplier = 1.0
                if nb_pos[2] < self.min_height:
                    # 如果低于阈值，大幅增加这一步的代价值
                    cost_multiplier = self.penalty
                
                tentative_g = cur_g + (step_dist * cost_multiplier)

                if tentative_g < g_score.get(nb, float('inf')):
                    g_score[nb] = tentative_g
                    f_score = tentative_g + h(nb)
                    came_from[nb] = current
                    heapq.heappush(open_heap, (f_score, tentative_g, nb))

        print('[AStarPlanner] warning: no path found')
        return None

    # ------------------------------------------------------------------ 路径处理

    @staticmethod
    def resample_path(path: np.ndarray, spacing: float = 0.5) -> np.ndarray:
        """对路径进行等间距重采样，适合 RL waypoint 或轨迹跟踪"""
        if path is None or len(path) < 2:
            return path
        
        seg = np.diff(path, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total_len = cum[-1]
        
        if total_len < 1e-6:
            return path[[0, -1]]
            
        n_points = max(2, int(np.ceil(total_len / spacing)) + 1)
        samples = np.linspace(0, total_len, n_points)
        
        resampled = []
        for s in samples:
            k = np.searchsorted(cum, s, side='right') - 1
            k = int(np.clip(k, 0, len(path) - 2))
            t = 0.0 if seg_len[k] < 1e-9 else (s - cum[k]) / seg_len[k]
            resampled.append(path[k] + t * (path[k + 1] - path[k]))
            
        resampled = np.array(resampled)
        resampled[-1] = path[-1] # 确保终点精确
        return resampled

# --- 使用示例 ---
if __name__ == "__main__":
    planner = AStarPlanner(min_height=1.0, low_alt_penalty=20.0)
    
    # 假设从地面 [0, 0, 0] 出发，去往 [3, 0, 1.5]
    start = [0.0, 0.0, 0.0]
    goal = [3.0, 0.0, 1.5]
    
    # 模拟一个障碍物挡在中间
    obs = [([1.5, 0.0, 1.2], 0.4)] # (center, radius)
    
    raw_path = planner.plan(start, goal, obs)
    if raw_path is not None:
        final_waypoints = planner.resample_path(raw_path, spacing=0.3)
        print(f"规划成功，生成的路径点数量: {len(final_waypoints)}")
        # 你会发现路径点会迅速爬升到 1.0m 以上，然后再水平移动