"""3D 栅格 A* 路径规划器。

障碍物用球体 (center_xyz, radius) 描述。规划空间是一个由 (x_range, y_range,
z_range) 与 grid_size 定义的均匀 3D 栅格。
"""

import heapq
import numpy as np


class AStarPlanner:
    def __init__(self,
                 grid_size: float = 0.2,
                 x_range=(-1.0, 5.0),
                 y_range=(-3.0, 3.0),
                 z_range=(0.1, 2.5),
                 safety_margin: float = 0.2):
        """
        Parameters
        ----------
        grid_size : float
            栅格分辨率 (米)。
        x_range, y_range, z_range : tuple(float, float)
            规划区域的最小/最大坐标。
        safety_margin : float
            无人机半径 / 安全余量, 用于障碍物膨胀。
        """
        self.grid_size = grid_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.safety_margin = safety_margin

        self.nx = max(1, int(np.ceil((self.x_max - self.x_min) / grid_size)))
        self.ny = max(1, int(np.ceil((self.y_max - self.y_min) / grid_size)))
        self.nz = max(1, int(np.ceil((self.z_max - self.z_min) / grid_size)))

    # ------------------------------------------------------------------ utils

    def _pos_to_index(self, pos):
        ix = int(round((pos[0] - self.x_min) / self.grid_size))
        iy = int(round((pos[1] - self.y_min) / self.grid_size))
        iz = int(round((pos[2] - self.z_min) / self.grid_size))
        ix = int(np.clip(ix, 0, self.nx - 1))
        iy = int(np.clip(iy, 0, self.ny - 1))
        iz = int(np.clip(iz, 0, self.nz - 1))
        return (ix, iy, iz)

    def _index_to_pos(self, idx):
        return np.array([self.x_min + idx[0] * self.grid_size,
                         self.y_min + idx[1] * self.grid_size,
                         self.z_min + idx[2] * self.grid_size])

    def _in_bounds(self, idx):
        return (0 <= idx[0] < self.nx and
                0 <= idx[1] < self.ny and
                0 <= idx[2] < self.nz)

    def _is_collision(self, idx, obstacles):
        """idx 处网格中心是否与任一障碍球相交 (考虑安全余量)."""
        pos = self._index_to_pos(idx)
        for (obs_pos, radius) in obstacles:
            if np.linalg.norm(pos - np.asarray(obs_pos)) < radius + self.safety_margin:
                return True
        return False

    # ------------------------------------------------------------------ plan

    def plan(self, start_pos, goal_pos, obstacles):
        """运行 A*, 返回 shape=(N,3) 的路径 (含起点与终点). 失败返回 None."""
        start_pos = np.asarray(start_pos, dtype=float)
        goal_pos = np.asarray(goal_pos, dtype=float)
        obstacles = obstacles if obstacles is not None else []

        start_idx = self._pos_to_index(start_pos)
        goal_idx = self._pos_to_index(goal_pos)

        # 26 邻居
        directions = [(dx, dy, dz)
                      for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1)
                      for dz in (-1, 0, 1)
                      if not (dx == 0 and dy == 0 and dz == 0)]

        def h(a, b):
            return np.linalg.norm(np.array(a) - np.array(b)) * self.grid_size

        open_heap = []
        heapq.heappush(open_heap, (h(start_idx, goal_idx), 0.0, start_idx))
        came_from = {start_idx: None}
        g_score = {start_idx: 0.0}

        while open_heap:
            _, cur_g, current = heapq.heappop(open_heap)
            if current == goal_idx:
                # 回溯
                path_idx = []
                node = current
                while node is not None:
                    path_idx.append(node)
                    node = came_from[node]
                path_idx.reverse()
                path = np.array([self._index_to_pos(i) for i in path_idx])
                # 用真实 start/goal 替换首尾, 保证精确
                path[0] = start_pos
                path[-1] = goal_pos
                return path

            for dx, dy, dz in directions:
                nb = (current[0] + dx, current[1] + dy, current[2] + dz)
                if not self._in_bounds(nb):
                    continue
                if self._is_collision(nb, obstacles):
                    continue
                step_cost = np.sqrt(dx * dx + dy * dy + dz * dz) * self.grid_size
                tentative = cur_g + step_cost
                if tentative < g_score.get(nb, np.inf):
                    g_score[nb] = tentative
                    f = tentative + h(nb, goal_idx)
                    came_from[nb] = current
                    heapq.heappush(open_heap, (f, tentative, nb))

        print('[AStarPlanner] warning: no path found')
        return None

    # ------------------------------------------------------------------ utils

    @staticmethod
    def resample_path(path: np.ndarray, spacing: float = 0.5) -> np.ndarray:
        """按弧长等距重采样, 保留首尾, 适合作为 RL waypoint 使用."""
        if path is None or len(path) < 2:
            return path
        seg = np.diff(path, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        total = cum[-1]
        if total < 1e-6:
            return path[[0, -1]]
        n = max(2, int(np.ceil(total / spacing)) + 1)
        samples = np.linspace(0, total, n)
        resampled = []
        for s in samples:
            k = np.searchsorted(cum, s, side='right') - 1
            k = int(np.clip(k, 0, len(path) - 2))
            t = 0.0 if seg_len[k] < 1e-9 else (s - cum[k]) / seg_len[k]
            resampled.append(path[k] + t * (path[k + 1] - path[k]))
        resampled = np.array(resampled)
        resampled[0] = path[0]
        resampled[-1] = path[-1]
        return resampled