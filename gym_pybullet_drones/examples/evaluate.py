"""评测框架: 比较 RL policy 与传统控制 baseline 在避障路径跟踪任务上的表现.

使用方法:
    # 在仓库根目录下运行
    python gym_pybullet_drones/examples/evaluate.py \
        --rl_model results/save-XX/best_model.zip \
        --episodes 30

设计目标:
    * 控制器统一接口 BaseFlightController, 方便接入新 baseline
    * 评测在同一组随机种子下生成的场景, 保证公平
    * 输出多维度指标 + 汇总表 + CSV
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Type

# --- 让脚本独立可运行 (与 learn_new.py 同样的 sys.path 处理) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
root_root = os.path.dirname(project_root)
for _p in (project_root, root_root):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np

from envs.GlobalPlannerAviary import GlobalPlannerAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


# ============================================================================
#                              控制器抽象接口
# ============================================================================

class BaseFlightController:
    """所有控制器统一接口. 子类只需实现 reset / act."""

    name: str = "base"

    def reset(self, env: GlobalPlannerAviary, obs: np.ndarray, info: dict) -> None:
        """每个 episode 开始时调用. 用于读取 waypoints / 重置内部状态."""
        pass

    def act(self,
            env: GlobalPlannerAviary,
            obs: np.ndarray,
            info: dict,
            t: float) -> np.ndarray:
        """根据当前观测产生动作 (shape = env.action_space.shape)."""
        raise NotImplementedError


# ----------------------------------------------------------------------------
#                              RL 控制器 (PPO)
# ----------------------------------------------------------------------------

class RLController(BaseFlightController):
    """加载 stable_baselines3 PPO 模型的控制器."""

    name = "rl_ppo"

    def __init__(self, model_path: str, deterministic: bool = True):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.name = f"rl_ppo({os.path.basename(model_path)})"

    def act(self, env, obs, info, t):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


# ----------------------------------------------------------------------------
#                  传统 baseline 1: A* + waypoint PID 跟踪
# ----------------------------------------------------------------------------

class WaypointPIDController(BaseFlightController):
    """直接跟踪环境内 A* 规划的 waypoints, 不用 RL.

    实现方式: 输出 action = waypoint_relative_position 作为 Δp (前 3 维),
    Δv 和 Δrpy 全 0. 因为环境里 _preprocessAction 把 Δp 加到当前位置作为 PID
    target_pos, 这等价于"用 DSLPIDControl 直奔下一个 waypoint".

    阈值与 GlobalPlannerAviary 一致 (ARRIVAL_RADIUS), 到达即切换下一个 wp.
    """

    name = "astar_pid"

    def __init__(self, action_clip: float = 1.0):
        self.action_clip = action_clip
        self._wp_idx = 0
        self._waypoints = None

    def reset(self, env, obs, info):
        self._wp_idx = 0
        self._waypoints = np.asarray(env.WAYPOINTS, dtype=np.float32)

    def act(self, env, obs, info, t):
        # 单 drone 假设
        cur_pos = env._getDroneStateVector(0)[0:3]
        # 切换 waypoint
        while self._wp_idx < len(self._waypoints) - 1:
            if np.linalg.norm(self._waypoints[self._wp_idx] - cur_pos) < env.ARRIVAL_RADIUS:
                self._wp_idx += 1
            else:
                break
        target = self._waypoints[self._wp_idx]
        delta_p = target - cur_pos
        # Δp 受动作空间限制
        delta_p = np.clip(delta_p, -self.action_clip, self.action_clip)
        action = np.zeros((env.NUM_DRONES, 9), dtype=np.float32)
        action[0, 0:3] = delta_p
        return action


# ----------------------------------------------------------------------------
#               传统 baseline 2: 直线 PID (忽略障碍, 直奔终点)
# ----------------------------------------------------------------------------

class StraightLinePIDController(BaseFlightController):
    """不用规划, 直接朝 GOAL 飞 (用于展示"没有避障规划"会怎么样)."""

    name = "straight_pid"

    def __init__(self, action_clip: float = 1.0):
        self.action_clip = action_clip

    def act(self, env, obs, info, t):
        cur_pos = env._getDroneStateVector(0)[0:3]
        delta_p = env.GOAL - cur_pos
        delta_p = np.clip(delta_p, -self.action_clip, self.action_clip)
        action = np.zeros((env.NUM_DRONES, 9), dtype=np.float32)
        action[0, 0:3] = delta_p
        return action


# ============================================================================
#                                 评测指标
# ============================================================================

@dataclass
class EpisodeMetrics:
    """单 episode 指标."""
    success: bool = False
    collided: bool = False
    out_of_bounds: bool = False
    truncated: bool = False
    waypoints_reached: int = 0
    num_waypoints: int = 0
    time_to_goal_s: float = float("nan")     # 仅成功时有效
    path_length_m: float = 0.0               # 实际飞行路径长度
    optimal_length_m: float = 0.0            # A* 路径几何长度
    path_efficiency: float = float("nan")     # optimal / actual, [0,1]
    min_obstacle_clearance_m: float = float("inf")  # 离最近障碍/边界最小距离
    mean_jerk: float = 0.0                   # 加速度变化率, 平滑度
    mean_speed: float = 0.0
    energy_proxy: float = 0.0                # ∑ rpm² (越小越省电)
    cumulative_reward: float = 0.0
    steps: int = 0


@dataclass
class AggregateMetrics:
    """跨 episode 汇总."""
    controller: str
    n_episodes: int
    success_rate: float
    collision_rate: float
    oob_rate: float
    truncate_rate: float
    mean_time_to_goal_s: float
    mean_path_length: float
    mean_path_efficiency: float
    mean_min_clearance: float
    mean_jerk: float
    mean_speed: float
    mean_energy: float
    mean_reward: float
    mean_steps: float
    raw: List[EpisodeMetrics] = field(default_factory=list, repr=False)


def _waypoint_total_length(env: GlobalPlannerAviary) -> float:
    pts = np.vstack([env.START.reshape(1, 3),
                     np.asarray(env.WAYPOINTS, dtype=float)])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _aggregate(name: str, eps: List[EpisodeMetrics]) -> AggregateMetrics:
    if not eps:
        raise ValueError("no episodes to aggregate")
    succ = [e for e in eps if e.success]

    def safe_mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    return AggregateMetrics(
        controller=name,
        n_episodes=len(eps),
        success_rate=sum(e.success for e in eps) / len(eps),
        collision_rate=sum(e.collided for e in eps) / len(eps),
        oob_rate=sum(e.out_of_bounds for e in eps) / len(eps),
        truncate_rate=sum(e.truncated for e in eps) / len(eps),
        mean_time_to_goal_s=safe_mean([e.time_to_goal_s for e in succ]),
        mean_path_length=safe_mean([e.path_length_m for e in succ]) if succ else float("nan"),
        mean_path_efficiency=safe_mean([e.path_efficiency for e in succ]) if succ else float("nan"),
        mean_min_clearance=safe_mean([e.min_obstacle_clearance_m for e in eps]),
        mean_jerk=safe_mean([e.mean_jerk for e in eps]),
        mean_speed=safe_mean([e.mean_speed for e in eps]),
        mean_energy=safe_mean([e.energy_proxy for e in eps]),
        mean_reward=safe_mean([e.cumulative_reward for e in eps]),
        mean_steps=safe_mean([e.steps for e in eps]),
        raw=eps,
    )


# ============================================================================
#                             单 episode 评测主循环
# ============================================================================

def run_one_episode(controller: BaseFlightController,
                    env_factory: Callable[[], GlobalPlannerAviary],
                    seed: int,
                    max_seconds: Optional[float] = None) -> EpisodeMetrics:
    """跑一个 episode, 返回一组指标. env 用完即关."""
    env = env_factory()
    obs, info = env.reset(seed=seed)
    controller.reset(env, obs, info)

    m = EpisodeMetrics()
    m.num_waypoints = int(env.num_waypoints)
    m.optimal_length_m = _waypoint_total_length(env)
    dt = env.CTRL_TIMESTEP

    prev_pos = env._getDroneStateVector(0)[0:3].copy()
    prev_vel = env._getDroneStateVector(0)[10:13].copy()
    prev_acc = np.zeros(3)

    max_steps = (env.EPISODE_LEN_SEC + 5) * env.CTRL_FREQ
    if max_seconds is not None:
        max_steps = min(max_steps, int(max_seconds * env.CTRL_FREQ))

    speeds, jerks, energies = [], [], []
    cum_r = 0.0
    terminated = truncated = False

    for step in range(max_steps):
        action = controller.act(env, obs, info,
                                 t=step / env.CTRL_FREQ)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_r += float(reward)
        m.steps += 1

        cur_pos = env._getDroneStateVector(0)[0:3].copy()
        cur_vel = env._getDroneStateVector(0)[10:13].copy()
        cur_acc = (cur_vel - prev_vel) / dt
        jerk = np.linalg.norm((cur_acc - prev_acc) / dt)
        jerks.append(jerk)
        speeds.append(float(np.linalg.norm(cur_vel)))

        # 路径长度增量
        m.path_length_m += float(np.linalg.norm(cur_pos - prev_pos))

        # 最近障碍/边界距离
        d_obs = env._min_obs_distance(cur_pos)
        m.min_obstacle_clearance_m = min(m.min_obstacle_clearance_m, float(d_obs))

        # 能耗 proxy: ∑ rpm² * dt (跟功率成正比)
        if env.last_clipped_action is not None:
            # last_clipped_action 是被 clip 后的 RPM (BaseAviary 内部存)
            try:
                rpms = np.asarray(env.last_clipped_action).reshape(-1)
                energies.append(float(np.sum(rpms ** 2)) * dt)
            except Exception:
                pass

        prev_pos = cur_pos
        prev_vel = cur_vel
        prev_acc = cur_acc

        if terminated or truncated:
            break

    # ----- 终止原因判定 -----
    last_pos = env._getDroneStateVector(0)[0:3]
    last_d_obs = env._min_obs_distance(last_pos)
    (xr, yr, zr) = env.WORKSPACE
    oob = (last_pos[0] < xr[0] or last_pos[0] > xr[1]
           or last_pos[1] < yr[0] or last_pos[1] > yr[1]
           or last_pos[2] < zr[0] or last_pos[2] > zr[1])
    final_to_goal = float(np.linalg.norm(env.WAYPOINTS[-1] - last_pos))
    reached_last = (env.wp_counters[0] >= env.num_waypoints - 1
                    and final_to_goal < env.ARRIVAL_RADIUS)

    m.collided = bool(last_d_obs < 0.0)
    m.out_of_bounds = bool(oob and not m.collided)
    m.success = bool(reached_last and not m.collided and not m.out_of_bounds)
    m.truncated = bool(truncated and not m.success)
    m.waypoints_reached = int(env.wp_counters[0]) + (1 if m.success else 0)
    m.cumulative_reward = cum_r
    m.mean_jerk = float(np.mean(jerks)) if jerks else 0.0
    m.mean_speed = float(np.mean(speeds)) if speeds else 0.0
    m.energy_proxy = float(np.sum(energies)) if energies else 0.0
    if m.success:
        m.time_to_goal_s = m.steps * dt
        if m.path_length_m > 1e-6:
            m.path_efficiency = m.optimal_length_m / m.path_length_m
    env.close()
    return m


# ============================================================================
#                              主评测入口
# ============================================================================

def evaluate_controller(controller: BaseFlightController,
                        env_factory: Callable[[], GlobalPlannerAviary],
                        seeds: List[int],
                        verbose: bool = True) -> AggregateMetrics:
    eps: List[EpisodeMetrics] = []
    for k, sd in enumerate(seeds):
        m = run_one_episode(controller, env_factory, seed=sd)
        eps.append(m)
        if verbose:
            print(f"  [{controller.name}] ep {k+1}/{len(seeds)} seed={sd} "
                  f"success={m.success} steps={m.steps} "
                  f"reward={m.cumulative_reward:.1f} "
                  f"clearance={m.min_obstacle_clearance_m:.3f}")
    return _aggregate(controller.name, eps)


def print_table(rows: List[AggregateMetrics]) -> None:
    headers = [
        "controller", "n", "succ%", "coll%", "oob%", "trunc%",
        "t_goal(s)", "path_len", "eff", "min_clr", "jerk", "speed", "energy", "reward"
    ]

    def fmt(x, p=2, pct=False):
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "  -  "
        if pct:
            return f"{x*100:.1f}"
        return f"{x:.{p}f}"

    table = []
    for r in rows:
        table.append([
            r.controller, str(r.n_episodes),
            fmt(r.success_rate, pct=True),
            fmt(r.collision_rate, pct=True),
            fmt(r.oob_rate, pct=True),
            fmt(r.truncate_rate, pct=True),
            fmt(r.mean_time_to_goal_s, 2),
            fmt(r.mean_path_length, 2),
            fmt(r.mean_path_efficiency, 3),
            fmt(r.mean_min_clearance, 3),
            fmt(r.mean_jerk, 1),
            fmt(r.mean_speed, 2),
            fmt(r.mean_energy, 1),
            fmt(r.mean_reward, 1),
        ])
    widths = [max(len(h), max(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep)
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def save_csv(path: str, rows: List[AggregateMetrics]) -> None:
    import csv
    keys = [k for k in asdict(rows[0]).keys() if k != "raw"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            d = asdict(r)
            w.writerow([d[k] for k in keys])


# ============================================================================
#                            controller registry
# ============================================================================

CONTROLLER_REGISTRY: Dict[str, Callable[[argparse.Namespace], BaseFlightController]] = {
    "rl":           lambda a: RLController(a.rl_model, deterministic=True),
    "astar_pid":    lambda a: WaypointPIDController(),
    "straight_pid": lambda a: StraightLinePIDController(),
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate controllers on GlobalPlannerAviary")
    parser.add_argument("--rl_model", type=str, default=None,
                        help="path to PPO .zip; required if 'rl' in --controllers")
    parser.add_argument("--controllers", nargs="+",
                        default=["rl", "astar_pid", "straight_pid"],
                        choices=list(CONTROLLER_REGISTRY.keys()))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed_base", type=int, default=1000)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional path to dump aggregate CSV")
    # 场景参数 (与 learn_new.py 默认一致)
    parser.add_argument("--start_center", nargs=3, type=float,
                        default=[0.0, 0.0, 1.0])
    parser.add_argument("--goal_center", nargs=3, type=float,
                        default=[3.5, 0.0, 1.0])
    parser.add_argument("--region_radius", type=float, default=0.3)
    parser.add_argument("--num_obstacles", type=int, default=2)
    args = parser.parse_args()

    if "rl" in args.controllers and not args.rl_model:
        parser.error("--rl_model is required when 'rl' is in --controllers")

    seeds = [args.seed_base + i for i in range(args.episodes)]

    # 同一份 env_factory, 保证不同 controller 在相同 seed 下场景一致
    def env_factory():
        return GlobalPlannerAviary(
            simple_scene=True,
            start_center=tuple(args.start_center),
            goal_center=tuple(args.goal_center),
            region_radius=args.region_radius,
            num_obstacles=args.num_obstacles,
            obs=ObservationType.KIN,
            act=ActionType.PID,
            gui=args.gui,
        )

    aggregates: List[AggregateMetrics] = []
    for cname in args.controllers:
        print(f"\n==== Evaluating controller: {cname} ====")
        ctrl = CONTROLLER_REGISTRY[cname](args)
        agg = evaluate_controller(ctrl, env_factory, seeds, verbose=True)
        aggregates.append(agg)

    print("\n==== Summary ====")
    print_table(aggregates)

    if args.csv:
        save_csv(args.csv, aggregates)
        print(f"\n[CSV] saved to {args.csv}")


if __name__ == "__main__":
    main()
