import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV
df = pd.read_csv("D:\homework\Grade3_2\drone\gym-pybullet-drones-main\\test_logs\\rpm_log_20260604_220418.csv")

# 选择某个无人机
drone0 = df[df["drone_id"] == 0]

# 时间轴（使用数据索引）
time = range(len(drone0))

# 画图
# plt.plot(time, drone0["cur_x"])
# plt.plot(time, drone0["cur_y"])
# plt.plot(time, drone0["cur_z"])
# plt.plot(time, drone0["target_x"], linestyle='--')
# plt.plot(time, drone0["target_y"], linestyle='--')
# plt.plot(time, drone0["target_z"], linestyle='--')

# plt.plot(time, drone0["vel_x"])
# plt.plot(time, drone0["vel_y"])
# plt.plot(time, drone0["vel_z"])
# plt.plot(time, drone0["vel_offset_x"], linestyle='--')
# plt.plot(time, drone0["vel_offset_y"], linestyle='--')
# plt.plot(time, drone0["vel_offset_z"], linestyle='--')
# plt.plot(time, drone0["vel_target_x"], linestyle='--')
# plt.plot(time, drone0["vel_target_y"], linestyle='--')
# plt.plot(time, drone0["vel_target_z"], linestyle='--')

# plt.plot(time, drone0["roll"])
# plt.plot(time, drone0["pitch"])
# plt.plot(time, drone0["yaw"])
# plt.plot(time, drone0["ang_offset_roll"], linestyle='--')
# plt.plot(time, drone0["ang_offset_pitch"], linestyle='--')
# plt.plot(time, drone0["ang_offset_yaw"], linestyle='--')

plt.plot(time, drone0["rpm_0"])
plt.plot(time, drone0["rpm_1"])
plt.plot(time, drone0["rpm_2"])
plt.plot(time, drone0["rpm_3"])
mean_rpm = (drone0["rpm_0"] + drone0["rpm_1"] + drone0["rpm_2"] + drone0["rpm_3"]) / 4
plt.plot(time, mean_rpm, linestyle='--')
delta0 = abs(drone0["rpm_0"] - mean_rpm)
delta1 = abs(drone0["rpm_1"] - mean_rpm)
delta2 = abs(drone0["rpm_2"] - mean_rpm)
delta3 = abs(drone0["rpm_3"] - mean_rpm)

max_delta_rpm = pd.concat(
    [delta0, delta1, delta2, delta3],
    axis=1
).max(axis=1)
plt.plot(time, max_delta_rpm, linestyle='-.', color='red')

plt.grid(True)

# plt.legend(["cur_x", "cur_y", "cur_z", "target_x", "target_y", "target_z"])

# plt.legend(["vel_x", "vel_y", "vel_z", "vel_offset_x", "vel_offset_y", "vel_offset_z", "vel_target_x", "vel_target_y", "vel_target_z"])

# plt.legend(["roll", "pitch", "yaw", "ang_offset_roll", "ang_offset_pitch", "ang_offset_yaw"])

plt.legend(["rpm_0", "rpm_1", "rpm_2", "rpm_3", "mean_rpm", "max_delta_rpm"])

plt.show()