import pandas as pd
import numpy as np
import glob
import os

# ====== 1. 路径 ======
folder = r"D:\\homework\Grade3_2\\drone\\gym-pybullet-drones-main\\test_logs"

start_file = "obs_clipped_log_20260605_153258.csv"
end_file   = "obs_clipped_log_20260605_153317.csv"

# ====== 2. 获取所有文件并排序 ======
files = sorted(glob.glob(os.path.join(folder, "obs_clipped_log_*.csv")))

# 截取区间（基于文件名字符串排序）
start_idx = files.index(os.path.join(folder, start_file))
end_idx   = files.index(os.path.join(folder, end_file))

selected_files = files[start_idx:end_idx + 1]

print(f"Selected {len(selected_files)} files")

# ====== 3. 读取并拼接 ======
dfs = []
for f in selected_files:
    df = pd.read_csv(f)
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# ====== 4. 统计 ======
cols = data.columns  # obs0 ~ obs14

stats = {}

for c in cols:
    x = data[c].values

    # 使用 pandas.Series 计算偏度和峰度
    s = pd.Series(x)

    stats[c] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skew": float(s.skew()),
        "kurtosis": float(s.kurt()),
        "p2": float(np.percentile(x, 2)),
        "p98": float(np.percentile(x, 98)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

# ====== 5. 转成表格 ======
stats_df = pd.DataFrame(stats).T

print(stats_df)

# ====== 6. 保存 ======
out_path = os.path.join(folder, "obs_stats.csv")
stats_df.to_csv(out_path)

print("Saved to:", out_path)

# 额外：将结果保存到 `gym_pybullet_drones/io`（基于脚本所在包目录）
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
io_dir = os.path.join(pkg_dir, "io")
os.makedirs(io_dir, exist_ok=True)

txt_path = os.path.join(io_dir, "obs_stats.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(stats_df.to_string())

print("Also saved text to:", txt_path)