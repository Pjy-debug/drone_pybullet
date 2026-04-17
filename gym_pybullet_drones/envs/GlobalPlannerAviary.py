IS_DEBUG = False

import numpy as np
from gymnasium import spaces
from envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class GlobalPlannerAviary(BaseRLAviary):
    """
    带有全局规划路径追踪功能的无人机强化学习环境。
    """
    def __init__(self,
                 waypoints: np.ndarray = None, # 形状为 (N, 3) 的中间点列表
                 arrival_radius: float = 0.15,  # 判定到达中间点的半径
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.EPISODE_LEN_SEC = 20

        # 初始化路径规划相关变量
        # 如果没有提供路径，默认原地踏步（实际应用中应由算法生成）
        self.WAYPOINTS = waypoints if waypoints is not None else np.array([0,0,1.5])
        print(f'[INFO] 使用的路径点: {self.WAYPOINTS}')
        self.num_waypoints = self.WAYPOINTS.shape[0]
        self.arrival_radius = arrival_radius
        
        # 每个无人机当前的路径点索引
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)
        
    def reset(self, seed=None, options=None):
        """重置环境并重置路径点进度"""
        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)
        return super().reset(seed=seed, options=options)

    def _observationSpace(self):
        """
        扩展观测空间：原有的 KIN 观测 (12) + 动作 Buffer + 当前目标点相对位置 (3)
        """
        obs_space = super()._observationSpace()
        if self.OBS_TYPE == ObservationType.KIN:
            # # 在原有的 Box 形状基础上，为每个无人机增加 3 维（相对于当前目标的 XYZ）
            # low = np.array([-np.inf] * 3)
            # high = np.array([np.inf] * 3)
            
            # 这里需要根据 BaseRLAviary 的实现重新构建 Box，
            # 简单起见，我们直接计算新的 shape
            current_shape = obs_space.shape # (NUM_DRONES, D)
            new_dim = current_shape[-1] + 3
            return spaces.Box(low=-np.inf, high=np.inf, shape=(self.NUM_DRONES, new_dim), dtype=np.float32)
        return obs_space

    def _computeObs(self):
        """
        计算观测：包含无人机状态和当前目标路径点信息
        """
        root_obs = super()._computeObs() # (NUM_DRONES, D)
        target_info = np.zeros((self.NUM_DRONES, 3))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            current_wp = self.WAYPOINTS[self.wp_counters[i]]
            # 使用相对位置有助于泛化
            target_info[i, :] = current_wp - state[0:3]
        if IS_DEBUG:
            print(f'Current waypoints: {self.WAYPOINTS[self.wp_counters]}')
            print(f'Current pos: {root_obs[:, 0:3]}')
            print(f'Current vel: {root_obs[:, 10:13]}')
            print(f'Target info (relative pos): {target_info}')
        return np.hstack([root_obs, target_info]).astype('float32')

    def _computeReward(self):
        """
        基于路径点追踪的奖励函数
        """
        rewards = np.zeros(self.NUM_DRONES)
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            pos = state[0:3]
            target_wp = self.WAYPOINTS[self.wp_counters[i]]
            vel = state[10:13]
            vel_normolized = vel / (np.linalg.norm(vel) + 1e-6)
            destination_vector = target_wp - pos
            dist = np.linalg.norm(destination_vector)
            destination_vector_normolized = destination_vector / (np.linalg.norm(destination_vector) + 1e-6)
            # 1. 方向奖励
            direction_reward = np.dot(vel_normolized, destination_vector_normolized)
            rewards[i] += (direction_reward > 0)*np.linalg.norm(vel)
            # 2. 到达中间点的奖励
            if dist < self.arrival_radius:
                if self.wp_counters[i] < self.num_waypoints - 1:
                    arrival_reward =  10.0  # 到达中间点奖励
                    self.wp_counters[i] += 1  # 切换到下一个目标
                else:
                    arrival_reward = 100.0 # 到达最终终点奖励
                rewards[i] += arrival_reward
            # 3. 碰撞惩罚（简单实现，BaseAviary 提供了碰撞检测接口）
            # if self._checkCollision(i): rewards[i] -= 50

            if IS_DEBUG:
                print(f'Drone {i}: 方向奖励 {direction_reward:.2f}, 到达奖励 {arrival_reward if dist < self.arrival_radius else 0:.2f}, 总奖励 {rewards[i]:.2f}')
        return rewards

    def _computeTerminated(self):
        """
        判定是否完成所有路径点
        """
        terminated = []
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            dist_to_final = np.linalg.norm(self.WAYPOINTS[-1] - state[0:3])
            
            # 如果到达最后一个点且速度较小，或者超过路径点索引
            done = self.wp_counters[i] == self.num_waypoints - 1 and dist_to_final < self.arrival_radius
            terminated.append(done)
            
        return all(terminated)

    def _computeTruncated(self):
        """
        超时或者飞出边界
        """
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC: 
            return True
        return False
    def _computeInfo(self):
        """
        计算当前的辅助信息。
        BaseAviary 要求必须实现此方法，否则会抛出 NotImplementedError。
        """
        # 返回一个包含每个无人机当前路径点进度的字典
        return {
            "answer": True, # 只是占位
            "current_wp_indices": self.wp_counters 
        }