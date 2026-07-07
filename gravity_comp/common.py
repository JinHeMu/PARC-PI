#!/usr/bin/env python3
"""
common.py — 六维力传感器重力/惯性补偿与参数辨识的公共模块（动态版）

相比静态版新增：
    compute_kinematic_derivatives()  ：按时间顺序算出平滑后的关节速度和关节加速度
    build_feature() 新增两组特征     ：关节加速度 qdd(6) 与向心项 qd^2(6)

动态刚体（Newton-Euler）下传感器测得：
    f   = m(a_s + w'xc + wx(wxc) - g) + f_bias      # 含向心(w^2)和加速度项
    tau = I w' + wx(I w) + m cx(a_s - g) + tau_bias
其中 a_s, w, w' 由 (q, qd, qdd) 通过雅可比决定，
因此网络输入应包含关节角 q、关节速度 qd、关节加速度 qdd。
"""
import os
import csv
import random

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim


CHANNELS = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
UNITS = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

# 默认特征开关：动态轨迹建议全开
DEFAULT_FEATURE_CFG = {
    'use_joint_pos': True,     # sin(q)/cos(q)   —— 关节构型
    'use_joint_vel': True,     # qd              —— 一阶动态
    'use_joint_acc': False,     # qdd             —— 加速度项（动态关键）
    'use_vel_products': False,  # qd^2            —— 向心项
}


# ============================================================
# 基础工具
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def skew_symmetric(v):
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


def rmse_np(error):
    return np.sqrt(np.mean(error ** 2, axis=0))


def _valid_savgol_window(window, n, poly):
    """返回合法的 Savitzky-Golay 窗口：奇数、>poly、<=n；不合法则返回 None。"""
    win = min(window, n)
    if win % 2 == 0:
        win -= 1
    if win < poly + 2:
        return None
    return win


# ============================================================
# 数据读取
# ============================================================
def load_csv(filename):
    """
    读取 CSV，返回 rows(list[dict]), has_joint_pos, has_joint_vel。
    rows 保持采集时间顺序（求导需要）。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find {filename}")

    rows = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        required_cols = ['qx', 'qy', 'qz', 'qw',
                         'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        for col in required_cols:
            if col not in fieldnames:
                raise RuntimeError(f"CSV missing required column: {col}")

        has_joint_pos = all(f'joint_{i}_pos' in fieldnames for i in range(1, 7))
        has_joint_vel = all(f'joint_{i}_vel' in fieldnames for i in range(1, 7))

        for row in reader:
            quat = np.array([float(row['qx']), float(row['qy']),
                             float(row['qz']), float(row['qw'])], dtype=np.float64)
            quat = quat / (np.linalg.norm(quat) + 1e-12)
            if quat[3] < 0:
                quat = -quat

            wrench = np.array([
                float(row['fx']), float(row['fy']), float(row['fz']),
                float(row['tx']), float(row['ty']), float(row['tz'])
            ], dtype=np.float64)

            q = np.array([float(row[f'joint_{i}_pos'])
                          for i in range(1, 7)], dtype=np.float64) if has_joint_pos else None
            qd = np.array([float(row[f'joint_{i}_vel'])
                           for i in range(1, 7)], dtype=np.float64) if has_joint_vel else None
            t = float(row['time_sec']) if 'time_sec' in row else None

            rows.append({'quat': quat, 'R': R.from_quat(quat).as_matrix(),
                         'wrench': wrench, 'q': q, 'qd': qd, 'time': t})

    return rows, has_joint_pos, has_joint_vel


def get_time_axis(rows):
    if all(r['time'] is not None for r in rows):
        t = np.array([r['time'] for r in rows], dtype=np.float64)
    else:
        t = np.arange(len(rows), dtype=np.float64)
    if len(t) > 0:
        t = t - t[0]
    return t


# ============================================================
# 关节速度平滑 + 关节加速度（动态特征的核心）
# ============================================================
def compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3,
                                  recompute_velocity=False):
    """
    按采集时间顺序计算：
        row['qd_feat'] : 用于特征的关节速度（可选平滑）
        row['qdd']     : 关节加速度（由速度数值微分 + 平滑）

    参数
    ----
    smooth              : 是否用 Savitzky-Golay 平滑（数值微分会放大噪声，建议开）
    window, poly        : 平滑窗口长度（奇数）与多项式阶
    recompute_velocity  : True 时用关节角数值微分得到速度（当 CSV 速度不可靠时）

    注意：必须在打乱/划分数据之前调用，因为求导依赖时间连续性。
    """
    n = len(rows)
    have_pos = n > 0 and rows[0]['q'] is not None
    have_vel = n > 0 and rows[0]['qd'] is not None

    if n < 3 or not (have_pos or have_vel):
        for r in rows:
            r['qd_feat'] = r['qd'] if r['qd'] is not None else np.zeros(6)
            r['qdd'] = np.zeros(6)
        return rows

    if all(r['time'] is not None for r in rows):
        t = np.array([r['time'] for r in rows], dtype=np.float64)
        dt = np.diff(t)
        if np.any(dt <= 0):
            step = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
            t = np.arange(n, dtype=np.float64) * step
    else:
        t = np.arange(n, dtype=np.float64)

    if recompute_velocity and have_pos:
        Q = np.vstack([r['q'] for r in rows])
        QD = np.gradient(Q, t, axis=0)
    elif have_vel:
        QD = np.vstack([r['qd'] for r in rows])
    else:
        Q = np.vstack([r['q'] for r in rows])
        QD = np.gradient(Q, t, axis=0)

    win = _valid_savgol_window(window, n, poly) if smooth else None
    QDs = savgol_filter(QD, win, poly, axis=0) if win else QD

    QDD = np.gradient(QDs, t, axis=0)
    if win:
        QDD = savgol_filter(QDD, win, poly, axis=0)

    for i, r in enumerate(rows):
        r['qd_feat'] = QDs[i]
        r['qdd'] = QDD[i]
    return rows


# ============================================================
# 方法 1：传统最小二乘物理辨识（重力模型）
# ============================================================
def identify_physics_params(rows):
    N = len(rows)
    recorded_R = [row['R'] for row in rows]
    recorded_F = [row['wrench'][0:3] for row in rows]
    recorded_T = [row['wrench'][3:6] for row in rows]

    A_force = np.zeros((3 * N, 6)); B_force = np.zeros((3 * N, 1))
    for i in range(N):
        A_force[i * 3:(i + 1) * 3, 0:3] = recorded_R[i]
        A_force[i * 3:(i + 1) * 3, 3:6] = np.eye(3)
        B_force[i * 3:(i + 1) * 3, 0] = recorded_F[i]
    X_force, res_F, rank_F, _ = np.linalg.lstsq(A_force, B_force, rcond=None)
    G, F_bias = X_force[0:3].flatten(), X_force[3:6].flatten()

    A_torque = np.zeros((3 * N, 6)); B_torque = np.zeros((3 * N, 1))
    for i in range(N):
        V = recorded_R[i] @ G
        A_torque[i * 3:(i + 1) * 3, 0:3] = -skew_symmetric(V)
        A_torque[i * 3:(i + 1) * 3, 3:6] = np.eye(3)
        B_torque[i * 3:(i + 1) * 3, 0] = recorded_T[i]
    X_torque, res_T, rank_T, _ = np.linalg.lstsq(A_torque, B_torque, rcond=None)
    CoM, T_bias = X_torque[0:3].flatten(), X_torque[3:6].flatten()

    return {'G': G, 'F_bias': F_bias, 'CoM': CoM, 'T_bias': T_bias,
            'mass': float(np.linalg.norm(G) / 9.81),
            'force_residuals': res_F, 'torque_residuals': res_T,
            'rank_F': rank_F, 'rank_T': rank_T}


def physics_predict_wrench(R_mat, params):
    G = params['G']
    F_pred = R_mat @ G + params['F_bias']
    T_pred = -skew_symmetric(R_mat @ G) @ params['CoM'] + params['T_bias']
    return np.concatenate([F_pred, T_pred])


def physics_predict_all(rows, params):
    return np.vstack([physics_predict_wrench(r['R'], params) for r in rows])


# ============================================================
# 神经网络输入特征（含动态项）
# ============================================================
def build_feature(row, params, cfg=None):
    """
    特征组成（按 cfg 开关）：
        3 维 : 传感器系单位重力方向
        4 维 : 四元数
        12维 : sin(q)/cos(q)         —— 关节构型
        6 维 : qd（平滑速度）         —— 一阶动态
        6 维 : qdd（关节加速度）      —— 加速度项 ★动态新增
        6 维 : qd^2（向心项）         —— 向心项   ★动态新增
    全开时维度 = 3+4+12+6+6+6 = 37。
    """
    if cfg is None:
        cfg = DEFAULT_FEATURE_CFG

    R_mat = row['R']
    g_sensor = R_mat @ params['G']
    g_sensor = g_sensor / (np.linalg.norm(g_sensor) + 1e-9)

    features = list(g_sensor)
    features.extend(row['quat'].tolist())

    if cfg['use_joint_pos'] and row['q'] is not None:
        features.extend(np.sin(row['q']).tolist())
        features.extend(np.cos(row['q']).tolist())

    qd = row.get('qd_feat', row['qd'])
    if cfg['use_joint_vel'] and qd is not None:
        features.extend(np.asarray(qd).tolist())

    if cfg['use_joint_acc'] and 'qdd' in row:
        features.extend(np.asarray(row['qdd']).tolist())

    if cfg['use_vel_products'] and qd is not None:
        features.extend((np.asarray(qd) ** 2).tolist())

    return np.array(features, dtype=np.float32)


def build_feature_matrix(rows, params, cfg=None):
    return np.vstack([build_feature(r, params, cfg) for r in rows]).astype(np.float32)


def feature_dim(cfg=None):
    if cfg is None:
        cfg = DEFAULT_FEATURE_CFG
    d = 3 + 4
    if cfg['use_joint_pos']:
        d += 12
    if cfg['use_joint_vel']:
        d += 6
    if cfg['use_joint_acc']:
        d += 6
    if cfg['use_vel_products']:
        d += 6
    return d


# ============================================================
# 神经网络
# ============================================================
class WrenchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def normalize_fit(X):
    return X.mean(axis=0), X.std(axis=0) + 1e-8


def train_network(X, Y, input_dim, hidden_dim=64,
                  epochs=2500, lr=1e-3, weight_decay=1e-4,
                  train_ratio=0.8, seed=42, verbose=True, tag=''):
    set_seed(seed)
    N = X.shape[0]
    idx = np.arange(N); np.random.shuffle(idx)
    ntr = int(N * train_ratio)
    tr, va = idx[:ntr], idx[ntr:]

    X_tr = torch.tensor(X[tr], dtype=torch.float32)
    Y_tr = torch.tensor(Y[tr], dtype=torch.float32)
    X_va = torch.tensor(X[va], dtype=torch.float32)
    Y_va = torch.tensor(Y[va], dtype=torch.float32)

    model = WrenchMLP(input_dim, hidden_dim, 6)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state = float('inf'), None
    if verbose:
        print(f"\n--- Training {tag} (input_dim={input_dim}) ---")
    for ep in range(1, epochs + 1):
        model.train()
        loss = loss_fn(model(X_tr), Y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_va), Y_va).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if verbose and (ep % 200 == 0 or ep == 1):
            print(f"  {tag} epoch {ep:4d} | train {loss.item():.6f} | val {vl:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_network(model, X_norm, y_mean, y_std):
    with torch.no_grad():
        out = model(torch.tensor(X_norm, dtype=torch.float32)).cpu().numpy()
    return out * y_std + y_mean
