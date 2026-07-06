#!/usr/bin/env python3
import os
import csv
import math
import random
import numpy as np

from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.optim as optim


def skew_symmetric(v):
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def load_csv(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find {filename}")

    rows = []

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        required_cols = ['qx', 'qy', 'qz', 'qw', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        for col in required_cols:
            if col not in fieldnames:
                raise RuntimeError(f"CSV missing required column: {col}")

        has_joint_pos = all(f'joint_{i}_pos' in fieldnames for i in range(1, 7))
        has_joint_vel = all(f'joint_{i}_vel' in fieldnames for i in range(1, 7))

        for row in reader:
            quat = np.array([
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            ], dtype=np.float64)

            # 四元数归一化
            quat = quat / np.linalg.norm(quat)

            # 避免 q 和 -q 跳变
            if quat[3] < 0:
                quat = -quat

            wrench = np.array([
                float(row['fx']),
                float(row['fy']),
                float(row['fz']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz'])
            ], dtype=np.float64)

            q = None
            qd = None

            if has_joint_pos:
                q = np.array([
                    float(row[f'joint_{i}_pos']) for i in range(1, 7)
                ], dtype=np.float64)

            if has_joint_vel:
                qd = np.array([
                    float(row[f'joint_{i}_vel']) for i in range(1, 7)
                ], dtype=np.float64)

            rows.append({
                'quat': quat,
                'R': R.from_quat(quat).as_matrix(),
                'wrench': wrench,
                'q': q,
                'qd': qd
            })

    return rows, has_joint_pos, has_joint_vel


def identify_physics_params(rows):
    N = len(rows)

    recorded_R = [row['R'] for row in rows]
    recorded_F = [row['wrench'][0:3] for row in rows]
    recorded_T = [row['wrench'][3:6] for row in rows]

    # ---------------------------------------------------------
    # 1. 辨识 G 和 F_bias
    # ---------------------------------------------------------
    A_force = np.zeros((3 * N, 6))
    B_force = np.zeros((3 * N, 1))

    for i in range(N):
        A_force[i*3:(i+1)*3, 0:3] = recorded_R[i]
        A_force[i*3:(i+1)*3, 3:6] = np.eye(3)
        B_force[i*3:(i+1)*3, 0] = recorded_F[i]

    X_force, residuals_F, rank_F, s_F = np.linalg.lstsq(
        A_force,
        B_force,
        rcond=None
    )

    G = X_force[0:3].flatten()
    F_bias = X_force[3:6].flatten()

    # ---------------------------------------------------------
    # 2. 辨识 CoM 和 T_bias
    # ---------------------------------------------------------
    A_torque = np.zeros((3 * N, 6))
    B_torque = np.zeros((3 * N, 1))

    for i in range(N):
        V = recorded_R[i] @ G
        V_skew = skew_symmetric(V)

        A_torque[i*3:(i+1)*3, 0:3] = -V_skew
        A_torque[i*3:(i+1)*3, 3:6] = np.eye(3)
        B_torque[i*3:(i+1)*3, 0] = recorded_T[i]

    X_torque, residuals_T, rank_T, s_T = np.linalg.lstsq(
        A_torque,
        B_torque,
        rcond=None
    )

    CoM = X_torque[0:3].flatten()
    T_bias = X_torque[3:6].flatten()

    params = {
        'G': G,
        'F_bias': F_bias,
        'CoM': CoM,
        'T_bias': T_bias,
        'force_residuals': residuals_F,
        'torque_residuals': residuals_T,
        'rank_F': rank_F,
        'rank_T': rank_T
    }

    return params


def physics_predict_wrench(R_mat, params):
    G = params['G']
    F_bias = params['F_bias']
    CoM = params['CoM']
    T_bias = params['T_bias']

    F_pred = R_mat @ G + F_bias

    V = R_mat @ G
    T_pred = -skew_symmetric(V) @ CoM + T_bias

    return np.concatenate([F_pred, T_pred])


def build_feature(row, params, use_joint_pos=True, use_joint_vel=True):
    R_mat = row['R']
    G = params['G']

    # 重力方向在传感器坐标系下的方向
    g_sensor = R_mat @ G
    g_sensor = g_sensor / (np.linalg.norm(g_sensor) + 1e-9)

    features = []

    # 最基础特征：重力方向
    features.extend(g_sensor.tolist())

    # 也加入四元数，增强网络表达能力
    quat = row['quat']
    features.extend(quat.tolist())

    # 加入关节角：用 sin/cos 避免角度 2π 跳变
    if use_joint_pos and row['q'] is not None:
        q = row['q']
        features.extend(np.sin(q).tolist())
        features.extend(np.cos(q).tolist())

    # 加入关节速度
    if use_joint_vel and row['qd'] is not None:
        qd = row['qd']
        features.extend(qd.tolist())

    return np.array(features, dtype=np.float32)


def make_dataset(rows, params, use_joint_pos=True, use_joint_vel=True):
    X = []
    Y = []

    for row in rows:
        wrench_meas = row['wrench']
        wrench_physics = physics_predict_wrench(row['R'], params)

        residual = wrench_meas - wrench_physics

        feature = build_feature(
            row,
            params,
            use_joint_pos=use_joint_pos,
            use_joint_vel=use_joint_vel
        )

        X.append(feature)
        Y.append(residual.astype(np.float32))

    X = np.vstack(X).astype(np.float32)
    Y = np.vstack(Y).astype(np.float32)

    return X, Y


def normalize_data(X, Y):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8

    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-8

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    return Xn, Yn, x_mean, x_std, y_mean, y_std


def rmse_np(error):
    return np.sqrt(np.mean(error ** 2, axis=0))


def main():
    filename = './datasets/payload_data1.csv'
    model_filename = './models/residual_compensator.pt'

    # 如果你新的文件名是这个，可以改成：
    # filename = 'payload_joint_wrench_quat_data.csv'

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows, has_joint_pos, has_joint_vel = load_csv(filename)

    print(f"Loaded {len(rows)} rows from {filename}")
    print(f"Has joint position: {has_joint_pos}")
    print(f"Has joint velocity: {has_joint_vel}")

    if len(rows) < 30:
        print("Warning: data is quite small. Residual network may overfit.")

    params = identify_physics_params(rows)

    G = params['G']
    F_bias = params['F_bias']
    CoM = params['CoM']
    T_bias = params['T_bias']
    mass = np.linalg.norm(G) / 9.81

    print("\n================ PHYSICS MODEL ================")
    print(f"Payload Mass (kg):      {mass:.4f}")
    print(f"G:                      {G}")
    print(f"CoM:                    {CoM}")
    print(f"F_bias:                 {F_bias}")
    print(f"T_bias:                 {T_bias}")
    print("==============================================\n")

    use_joint_pos = has_joint_pos
    use_joint_vel = has_joint_vel

    X, Y = make_dataset(
        rows,
        params,
        use_joint_pos=use_joint_pos,
        use_joint_vel=use_joint_vel
    )

    print(f"Input feature dimension: {X.shape[1]}")
    print(f"Output residual dimension: {Y.shape[1]}")

    # 物理模型残差 RMSE，也就是网络训练前的误差
    print("\nResidual before NN compensation:")
    before_rmse = rmse_np(Y)
    print(f"Fx,Fy,Fz RMSE: {before_rmse[0]:.4f}, {before_rmse[1]:.4f}, {before_rmse[2]:.4f} N")
    print(f"Tx,Ty,Tz RMSE: {before_rmse[3]:.4f}, {before_rmse[4]:.4f}, {before_rmse[5]:.4f} Nm")

    Xn, Yn, x_mean, x_std, y_mean, y_std = normalize_data(X, Y)

    N = Xn.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_ratio = 0.8
    train_size = int(N * train_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    X_train = torch.tensor(Xn[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Yn[train_idx], dtype=torch.float32)
    X_val = torch.tensor(Xn[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Yn[val_idx], dtype=torch.float32)

    input_dim = X.shape[1]
    model = ResidualMLP(input_dim=input_dim, hidden_dim=64, output_dim=6)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    epochs = 2000

    print("\n--- Training residual network ---")

    for epoch in range(1, epochs + 1):
        model.train()

        pred = model(X_train)
        loss = loss_fn(pred, Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"train loss: {loss.item():.6f} | "
                f"val loss: {val_loss.item():.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------------------------------------------------
    # 评估补偿效果
    # ---------------------------------------------------------
    model.eval()

    with torch.no_grad():
        X_all = torch.tensor(Xn, dtype=torch.float32)
        pred_residual_norm = model(X_all).cpu().numpy()

    pred_residual = pred_residual_norm * y_std + y_mean

    after_error = Y - pred_residual

    after_rmse = rmse_np(after_error)

    print("\nResidual after NN compensation:")
    print(f"Fx,Fy,Fz RMSE: {after_rmse[0]:.4f}, {after_rmse[1]:.4f}, {after_rmse[2]:.4f} N")
    print(f"Tx,Ty,Tz RMSE: {after_rmse[3]:.4f}, {after_rmse[4]:.4f}, {after_rmse[5]:.4f} Nm")

    # ---------------------------------------------------------
    # 保存模型
    # ---------------------------------------------------------
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': 64,

        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std,

        'G': G,
        'F_bias': F_bias,
        'CoM': CoM,
        'T_bias': T_bias,

        'use_joint_pos': use_joint_pos,
        'use_joint_vel': use_joint_vel
    }

    torch.save(checkpoint, model_filename)

    print(f"\nSaved residual compensator to: {model_filename}")


if __name__ == '__main__':
    main()
