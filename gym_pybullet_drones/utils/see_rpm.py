import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 配置
# =========================

log_dir = Path(r"D:\homework\Grade3_2\drone\gym-pybullet-drones-main\test_logs")

start_name = "rpm_log_20260604_220415.csv"
end_name   = "rpm_log_20260604_220509.csv"

save_dir = log_dir / "figures"
save_dir.mkdir(exist_ok=True)

dist_dir = log_dir / "statistics"
dist_dir.mkdir(exist_ok=True)

# =========================
# 全局收集
# =========================

all_rpm = []
all_mean_rpm = []
all_max_delta = []
all_ratio = []

all_collective = []
all_roll = []
all_pitch = []
all_yaw = []

# =========================
# 遍历文件
# =========================

files = sorted(log_dir.glob("rpm_log_*.csv"))

for file in files:

    name = file.name

    if name < start_name or name > end_name:
        continue

    print(f"Processing: {name}")

    df = pd.read_csv(file)
    drone0 = df[df["drone_id"] == 0]

    if len(drone0) == 0:
        continue

    time = range(len(drone0))

    rpm_cols = ["rpm_0", "rpm_1", "rpm_2", "rpm_3"]

    rpm0 = drone0["rpm_0"]
    rpm1 = drone0["rpm_1"]
    rpm2 = drone0["rpm_2"]
    rpm3 = drone0["rpm_3"]

    # =========================
    # mean / delta
    # =========================

    mean_rpm = drone0[rpm_cols].mean(axis=1)

    max_delta_rpm = (
        drone0[rpm_cols]
        .sub(mean_rpm, axis=0)
        .abs()
        .max(axis=1)
    )

    ratio = max_delta_rpm / mean_rpm

    # =========================
    # control space transform
    # =========================

    collective = (rpm0 + rpm1 + rpm2 + rpm3) / 4

    roll  = (-rpm0 - rpm1 + rpm2 + rpm3) / 4
    pitch = (-rpm0 + rpm1 + rpm2 - rpm3) / 4
    yaw   = (-rpm0 + rpm1 - rpm2 + rpm3) / 4

    # =========================
    # collect global stats
    # =========================

    for col in rpm_cols:
        all_rpm.extend(drone0[col].values)

    all_mean_rpm.extend(mean_rpm.values)
    all_max_delta.extend(max_delta_rpm.values)
    all_ratio.extend(ratio.values)

    all_collective.extend(collective.values)
    all_roll.extend(roll.values)
    all_pitch.extend(pitch.values)
    all_yaw.extend(yaw.values)

    # =========================
    # plot per file
    # =========================

    plt.figure(figsize=(10, 5))

    plt.plot(time, rpm0)
    plt.plot(time, rpm1)
    plt.plot(time, rpm2)
    plt.plot(time, rpm3)

    plt.plot(time, mean_rpm, "--")
    plt.plot(time, max_delta_rpm, "-.", color="red")

    plt.title(name)

    plt.legend([
        "rpm_0",
        "rpm_1",
        "rpm_2",
        "rpm_3",
        "mean_rpm",
        "max_delta_rpm"
    ])

    plt.grid(True)

    plt.savefig(save_dir / f"{file.stem}.png", dpi=200, bbox_inches="tight")
    plt.close()

# =========================
# to numpy
# =========================

all_rpm = np.array(all_rpm)
all_mean_rpm = np.array(all_mean_rpm)
all_max_delta = np.array(all_max_delta)
all_ratio = np.array(all_ratio)

all_collective = np.array(all_collective)
all_roll = np.array(all_roll)
all_pitch = np.array(all_pitch)
all_yaw = np.array(all_yaw)

# =========================
# stats function
# =========================

def print_stats(name, data):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"count : {len(data)}")
    print(f"min   : {data.min():.3f}")
    print(f"max   : {data.max():.3f}")
    print(f"mean  : {data.mean():.3f}")
    print(f"std   : {data.std():.3f}")

    print(f"p1    : {np.percentile(data, 1):.3f}")
    print(f"p5    : {np.percentile(data, 5):.3f}")
    print(f"p50   : {np.percentile(data, 50):.3f}")
    print(f"p95   : {np.percentile(data, 95):.3f}")
    print(f"p99   : {np.percentile(data, 99):.3f}")

# =========================
# print stats
# =========================

print_stats("RPM", all_rpm)
print_stats("Mean RPM", all_mean_rpm)
print_stats("Max Delta RPM", all_max_delta)
print_stats("Delta Ratio", all_ratio)

print_stats("Collective", all_collective)
print_stats("Roll", all_roll)
print_stats("Pitch", all_pitch)
print_stats("Yaw", all_yaw)

# =========================
# histogram helper
# =========================

def plot_hist(data, title, filename):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=100, density=True)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("Density")
    plt.grid(True)

    plt.savefig(dist_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# distributions
# =========================

plot_hist(all_rpm, "RPM Distribution", "rpm_distribution.png")
plot_hist(all_mean_rpm, "Mean RPM Distribution", "mean_rpm_distribution.png")
plot_hist(all_max_delta, "Max Delta RPM Distribution", "max_delta_distribution.png")
plot_hist(all_ratio, "Relative Delta Distribution", "relative_delta_distribution.png")

plot_hist(all_collective, "Collective Distribution", "collective_distribution.png")
plot_hist(all_roll, "Roll Distribution", "roll_distribution.png")
plot_hist(all_pitch, "Pitch Distribution", "pitch_distribution.png")
plot_hist(all_yaw, "Yaw Distribution", "yaw_distribution.png")

print("\nDone.")
print(f"Curve figures saved to: {save_dir}")
print(f"Statistics figures saved to: {dist_dir}")