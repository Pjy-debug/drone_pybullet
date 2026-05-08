# Drone RL 设计笔记：State / Action / DSLPIDControl

> 面向"懂物理但不熟自动控制"的读者。所有内容来自当前仓库实现：
> [`GlobalPlannerAviary.py`](../gym_pybullet_drones/envs/GlobalPlannerAviary.py) ·
> [`BaseRLAviary.py`](../gym_pybullet_drones/envs/BaseRLAviary.py) ·
> [`DSLPIDControl.py`](../gym_pybullet_drones/control/DSLPIDControl.py)

---

## 1. 观测 State（33 维 / 单架无人机）

观测在 [`GlobalPlannerAviary._computeObs`](../gym_pybullet_drones/envs/GlobalPlannerAviary.py) 里组装，结构为
`shape = (NUM_DRONES, 33)`，`float32`，未归一化（米、米/秒、弧度、弧度/秒）。

| 段 | 维数 | 含义 | 来源 |
|---|---|---|---|
| 位置 | 3 | x, y, z (世界坐标系) | `state[0:3]` |
| 姿态欧拉角 | 3 | roll, pitch, yaw | `state[7:10]` |
| 线速度 | 3 | vx, vy, vz | `state[10:13]` |
| 角速度 | 3 | wx, wy, wz | `state[13:16]` |
| Action buffer | 2 × 9 = 18 | 最近 2 步动作（每步 9 维） | `BaseRLAviary.action_buffer`，`ACTION_BUFFER_SIZE=2` |
| **当前 waypoint 相对位置** | 3 | `WAYPOINTS[wp_counter] - pos` | GlobalPlannerAviary 自定义追加 |

**合计 12 + 18 + 3 = 33 维。**

### 关键事实
1. **不含障碍物信息**：观测中没有任何障碍位置/距离/雷达/视觉数据。
2. **避障通过 A\* 间接实现**：障碍信息已经"烘焙"进 waypoints 序列。policy 只跟踪 `wp - pos`。
3. **没有起点/终点的绝对坐标**：policy 只看到当前 waypoint 的相对位移。
4. **没有四元数**，只有欧拉角 (roll, pitch, yaw)；没有原始 RPM。
5. KIN 观测 **没有归一化**（`_clipAndNormalizeState` 被注释掉了）。

> 想让 policy 真正"看到"障碍：在 `_computeObs` 里追加最近 K 个障碍 `(center - pos, radius)`，或用 `pybullet.rayTestBatch` 模拟一圈伪激光雷达。

---

## 2. 动作 Action（9 维 / 单架无人机）

动作定义在 [`GlobalPlannerAviary._actionSpace` / `_preprocessAction`](../gym_pybullet_drones/envs/GlobalPlannerAviary.py)。

| 维度 | 含义 | 单位 | low | high |
|---|---|---|---|---|
| 0–2 | Δposition (Δx, Δy, Δz) | m | -1.0 | 1.0 |
| 3–5 | Δvelocity (Δvx, Δvy, Δvz) | m/s | -0.25 | 0.25 |
| 6–8 | Δattitude (Δroll, Δpitch, Δyaw) | rad | -1.0 | 1.0 |

### 含义：相对增量，叠加到当前状态作为 PID 目标
```python
target_pos = action[0:3] + cur_pos        # 目标位置 = 当前位置 + Δp
target_vel = action[3:6] + cur_vel        # 目标速度 = 当前速度 + Δv
target_rpy = action[6:9] + cur_rpy        # 目标姿态 = 当前姿态 + Δrpy
```

- 然后用 `_calculateNextStep` 在 `target_pos` 方向上推一小步得到 `next_pos`。
- `next_pos / target_vel / target_rpy` 一起喂给 `DSLPIDControl.computeControl`。
- PID 算出 4 个螺旋桨的 RPM，由 PyBullet 实际驱动。

### 关键事实
1. **不是直接 RPM 控制**：动作经过 PID，物理可行且平滑。
2. **是相对增量**：输出 0 ≈ 悬停（保持当前位姿/速度）。
3. **action buffer 反馈进观测**：最近 2 步的 9 维动作拼到下一次 obs 里（共 18 维），让 policy 看到自己刚才输出。
4. **理论上 9 维冗余**：单纯追踪 waypoint 只需 Δposition (3 维) 即可。Δvel / Δrpy 给 agent 更细粒度控制，也增大探索空间。

---

## 3. DSLPIDControl 工作原理（大白话版）

### 3.1 为什么需要"分两层"控制？

四旋翼是 **欠驱动** 的：4 个电机只能产生「沿机身 z 轴的总推力」 + 「3 个轴的力矩」共 4 个独立量。但我们想控制的是 6 个自由度（位置 xyz + 姿态 rpy）。

物理上的事实是：**飞机只能通过倾斜机身把推力分量"歪"到水平方向，才能水平加速**。

> 想"往右飞"时，必须先"右倾"，让原本竖直向上的推力分到一部分向右；然后水平方向就有加速度。

所以工程上拆成两层：

| 控制环 | 频率 | 决定什么 | 输出给谁 |
|---|---|---|---|
| **位置环（外环）** | 慢 | 我**想去哪、走多快** → 推算"应该往哪个方向加速、加速多少" → **机身要倾成什么姿势 + 总推力多大** | 内环 |
| **姿态环（内环）** | 快 | 我**机身现在歪了多少、还差多少** → 算出 3 轴上"要给多大的力矩去拧" | 电机混控 |
| **混控（Mixer）** | — | 把"总推力 + 3 轴力矩"分配给 4 个电机 → 4 个 RPM | 物理 |

简而言之：**位置环把"飞到哪"翻译成"摆什么姿势 + 加多大油门"，姿态环负责像调平台秤一样把姿势调到那个值**。

### 3.2 PID 是什么（一句话版本）

PID 是一种"误差驱动"的控制器，核心想法极其朴素：

> **看现在差多少（P）、过去累计差了多少（I）、最近差距变化得多快（D），三者加权后输出一个"修正量"。**

- **P (Proportional)**：当前误差越大，修得越猛。像弹簧——离平衡点远，回拉力大。
- **I (Integral)**：过去一段时间误差累积起来。专门解决"长期偏一点点回不去"的稳态偏差（比如重力没补够导致一直差几厘米）。
- **D (Derivative)**：误差变化率。本质是"减震"——快冲过头时给反向力，避免震荡。

类比物理：**PID 控制 ≈ 弹簧 (P) + 累积力 (I) + 阻尼 (D)**。

### 3.3 位置环 `_dslPIDPositionControl`：把"去哪"翻译成"摆什么姿势 + 多大油门"

输入：当前 `pos, vel, quat`，目标 `target_pos, target_vel, target_yaw`。
做了 5 件事：

**Step 1 — 算误差**
- 位置误差 $e_p = p^* - p$（差多远）
- 速度误差 $e_v = v^* - v$（速度差多少）

**Step 2 — 攒积分**
$I \mathrel{+}= e_p \cdot dt$，并 clip 防"积太多冲过头"（anti-windup）。

**Step 3 — 算"想要的加速度方向"**
这是位置环的灵魂一步：
$$a^{\text{desired}} = K_P \odot e_p + K_I \odot I + K_D \odot e_v + (0, 0, g)$$

直觉解释：
- 前三项是 PID 修正：哪里偏了就往哪推。
- 最后加上 $(0, 0, g)$ 是 **重力补偿** —— 因为飞机自身重力 mg 一直在拽它下来，控制器必须先抵掉这个，剩下的才是"额外想要的加速度"。

> $a^{\text{desired}}$ 这个向量就是"我现在想让飞机往哪个方向、多大力地推"。

**Step 4 — 把"想要的加速度"翻译成"机身姿势 + 推力大小"**

物理事实：四旋翼只能沿机身 z 轴产生推力。所以为了让推力方向 = $a^{\text{desired}}$ 方向，**机身的 z 轴必须对齐 $a^{\text{desired}}$**。

具体做法：
- 目标机身 z 轴：$z^* = a^{\text{desired}} / \|a^{\text{desired}}\|$
- 给定一个期望 yaw（绕竖直方向旋转），用 yaw 和 $z^*$ 唯一确定 $x^*, y^*$（参考向量法）
- 拼成 `target_rotation` → 再转成 `target_euler (roll*, pitch*, yaw*)`

类比：你想斜着推一根棍子，棍子只能沿自己长度方向产生力 → 那就把棍子旋到那个方向再推。位置环要求的"机身姿势"就是这么来的。

**Step 5 — 算总推力**
推力大小 = $a^{\text{desired}}$ 在当前机身 z 轴方向上的投影：
$$\text{thrust} = \max(0,\ a^{\text{desired}} \cdot z_{\text{cur}})$$
再换算成 PWM 数值（4 个电机平均出力的等效 PWM）。

> 注意是投影到 **当前**机身 z（不是目标），这样在姿态还没调到位时也不会爆推。

**位置环输出**：`thrust`（总推力 PWM）和 `target_euler`（机身要摆成的姿势）。

### 3.4 姿态环 `_dslPIDAttitudeControl`：把"想要的姿势"变成"3 轴扭矩"

这一步的目标：让机身的旋转矩阵 $R$ 追踪目标 $R_t$。

**Step 1 — 旋转误差**

不能直接用欧拉角作差（万向锁、跨界等问题），而是用旋转矩阵之差的反对称部分：
$$E = R_t^\top R - R^\top R_t,\quad e_R = (E_{2,1}, E_{0,2}, E_{1,0})^\top$$

直觉：$e_R$ 大致告诉你"绕哪根轴、转多少"才能把当前姿态拧到目标姿态。

**Step 2 — 角速度误差**

通过差分估算当前角速度（拿 rpy 做一阶差分），再 $e_\omega = \omega^* - \dot\omega$。

**Step 3 — PID 算 3 轴目标力矩**
$$\tau^* = -K_P^{\text{att}} \odot e_R + K_D^{\text{att}} \odot e_\omega + K_I^{\text{att}} \odot I_R$$
- 比例：误差大 → 力矩大（像弹簧把姿态拽回去）
- 微分：转得太快 → 反向力矩（阻尼，防震荡）
- 积分：纠长期偏差（比如螺旋桨不对称带来的固定偏置）

`K_P^{\text{att}} ≈ 7×10⁴`，量级远大于位置环的 0.4 —— 这反映了「内环要比外环响应更快、更硬」的工程原则。

### 3.5 混控（Mixer）：把"总推力 + 3 轴力矩"分给 4 个电机

四旋翼有 4 个电机，每个电机产生独立的"上推力 + 自旋反扭矩"。`MIXER_MATRIX` 是一个 4×3 矩阵，告诉你「想要多少滚转/俯仰/偏航力矩 → 4 个电机分别加/减多少出力」：

```python
pwm_i = thrust + (MIXER_MATRIX @ τ*)[i]   # i = 0..3
pwm_i = clip(pwm_i, 20000, 65535)
rpm_i = 0.2685 * pwm_i + 4070.3            # PWM → RPM 的硬件标定
```

CF2X 模式（X 字形布局）的混控矩阵：
```
[-0.5, -0.5, -1]   →  螺旋桨 0
[-0.5,  0.5,  1]   →  螺旋桨 1
[ 0.5,  0.5, -1]   →  螺旋桨 2
[ 0.5, -0.5,  1]   →  螺旋桨 3
```

直觉：
- **滚转力矩** (绕 x) 通过让左右两侧电机出力不一样实现
- **俯仰力矩** (绕 y) 通过前后两侧电机出力不一样实现
- **偏航力矩** (绕 z) 通过对角电机的旋转反扭矩之差实现（CW/CCW 螺旋桨交替排布）

### 3.6 整体数据流总览

```
RL action (9D)
     │
     ▼  _preprocessAction
target_pos, target_vel, target_rpy
     │
     ▼  computeControl
┌──────────────── DSLPIDControl ─────────────────┐
│                                                │
│  位置环 PID                                     │
│   ├ 算 e_p, e_v, I                             │
│   ├ a_desired = PID + (0,0,g)                  │
│   ├ thrust = a_desired · z_body  (投影到机身z) │
│   └ target_R = (a_desired方向, 期望yaw)        │
│                                                │
│  姿态环 PID                                     │
│   ├ e_R = R_t^T R - R^T R_t (反对称提取)       │
│   ├ e_ω = ω* - ω̂ (差分估计当前角速度)         │
│   └ τ* = -P·e_R + D·e_ω + I·I_R                │
│                                                │
│  Mixer                                         │
│   pwm_i = thrust + (M·τ*)_i, clip → rpm_i      │
└────────────────────────────────────────────────┘
     │
     ▼
4× RPM → PyBullet 物理模拟
```

### 3.7 在 RL 训练中的角色

[`GlobalPlannerAviary._preprocessAction`](../gym_pybullet_drones/envs/GlobalPlannerAviary.py) 把 RL 输出的 9 维"高层增量"变成 PID 的目标：

```python
target_pos = action[0:3] + cur_pos
target_vel = action[3:6] + cur_vel
target_rpy = action[6:9] + cur_rpy
rpm = ctrl.computeControl(...,
    target_pos=next_pos,  # 经 _calculateNextStep 限步长
    target_rpy=target_rpy,
    target_vel=target_vel)
```

**含义：RL 只输出"高层目标偏置"，底层闭环稳定由 PID 负责。** 这极大简化了 agent 学习的问题：
- agent **不用学姿态稳定**（PID 内环搞定）
- agent **不用学重力补偿**（位置环包含 $(0,0,g)$）
- agent 只需学"路径跟踪 + 避障"这种高层任务

这是为什么这个任务在简化场景下很容易学的根本原因。

### 3.8 关键参数（来自 [`BaseControl`](../gym_pybullet_drones/control/BaseControl.py) 加载的 URDF）

- `KF`：单电机推力系数 (N·s²)
- `KM`：单电机扭矩系数
- `GRAVITY = M·g`：飞机总重力
- `MAX_RPM, HOVER_RPM`：电机硬件上下限

`computeControl` 内的所有数值常数都是**针对 Crazyflie 2.x 调过的**，所以代码里强制 `DRONE_MODEL` 必须是 `CF2X` 或 `CF2P`，否则直接 `exit()`。
