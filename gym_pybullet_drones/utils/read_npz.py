import numpy as np

# 替换为你实际的文件路径
path = 'D:\\homework\\Grade3_2\\drone\\gym-pybullet-drones-main\\gym_pybullet_drones\\results\\save-04.18.2026_18.10.18\\evaluations.npz'
data = np.load(path)

print("文件包含的键:", data.files)
# 通常包含: 'timesteps', 'results', 'ep_lengths', 'is_best'

timesteps = data['timesteps']  # 发生评估时的训练步数
results = data['results']      # 评估时的奖励（形状通常是 [评估次数, 评估回合数]）
ep_lengths = data['ep_lengths']  # 每次评估的回合长度
# 计算每次评估的平均奖励
print(f"Step: {timesteps}, results:{results}, Episode Length: {ep_lengths}")