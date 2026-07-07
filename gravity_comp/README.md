# 六维力传感器重力/惯性补偿 —— 三方法对比（动态版）

对机械臂末端六维力/力矩传感器做补偿与参数辨识，比较三种方法：

| 编号 | 方法 | 预测模型 | 补偿后残差 |
|------|------|----------|------------|
| 1 | 传统方法 | 物理最小二乘 `F=R·G+F_bias`, `T=-skew(R·G)·CoM+T_bias` | `meas − physics` |
| 2 | 传统神经网络 | MLP 直接回归 wrench | `meas − MLP(feat)` |
| 3 | 残差神经网络 | 物理模型 + MLP 补残差 | `meas − physics − MLP(feat)` |

三种方法在**同一数据、同一随机 80/20 划分**下比较。补偿后残差越接近 0 越好。

## 动态轨迹的处理（重点）

运动中传感器不仅测重力，还测负载的**惯性力/力矩**（Newton-Euler）：

```
f   = m(a_s + w'×c + w×(w×c) − g) + f_bias
tau = I·w' + w×(I·w) + m·c×(a_s − g) + tau_bias
```

- `w×(w×c)` 向心项 ∝ 速度平方 → 需要关节速度 `qd`
- `a_s, w'` 加速度项 ∝ `qdd` → **需要关节加速度**
- `a_s,w,w'` 经雅可比与 `q,qd,qdd` 相关 → 需要关节角 `q`

因此网络特征包含 `(q, qd, qdd, qd²)`。加速度由关节速度**数值微分 + Savitzky-Golay 平滑**离线得到（`compute_kinematic_derivatives`），CSV 无需改动。

特征开关见 `common.py::DEFAULT_FEATURE_CFG`：

```python
{'use_joint_pos': True,     # sin(q)/cos(q)  12 维
 'use_joint_vel': True,     # qd             6 维
 'use_joint_acc': True,     # qdd            6 维  ★动态关键
 'use_vel_products': True}  # qd²            6 维  ★向心项
```

全开时特征维度 = 3(重力方向)+4(四元数)+12+6+6+6 = **37**。

> 注：传统重力法要扩展到动态需要机器人运动学/雅可比（DH 参数）才能算 `a_s,w,w'`；
> 神经网络直接从 `(q,qd,qdd)` 学映射，**绕过了对完整运动学模型的依赖**，这是动态场景下 NN 的优势。

## 目录结构

```
gravity_comp/
├── common.py                    # 公共模块：读取、求导、物理辨识、特征、MLP、训练
├── data_collector.py            # ROS2 采集节点
├── offline_identifier.py        # 方法1：传统最小二乘（独立，无需 torch）
├── train_traditional_net.py     # 方法2：传统神经网络（含 qdd/qd²）
├── train_residual_net.py        # 方法3：残差神经网络（含 qdd/qd²）
├── compare_three_methods.py     # ★ 一键三方法对比 + 出图（主入口）
├── datasets/data2.csv
├── models/
└── imgs/
```

## 运行

```bash
pip install numpy scipy matplotlib torch
cd gravity_comp
python compare_three_methods.py     # 一键对比出图
```

## 参数说明

- `compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3, recompute_velocity=False)`
  - `window/poly`：平滑窗口与阶数。运动越快、采样越高可适当调整。
  - `recompute_velocity=True`：当 CSV 的 `*_vel` 不可靠时，用关节角数值微分得到速度。
- 想做消融（是否加加速度），把 `cfg` 中对应开关设为 `False` 再跑即可。

## 本数据集上的实测提示

`data2.csv` 关节速度很慢（≤0.07 rad/s），惯性项极小，
加入 `qdd/qd²` 对 RMSE 几乎无改善；当采集更快、更动态的轨迹时这些特征才会显著起作用。
代码已默认开启，无需改动即可适配更动态的数据。
