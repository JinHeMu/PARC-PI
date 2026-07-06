#!/usr/bin/env python3
import csv
import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


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


def physics_predict_wrench(R_mat, G, F_bias, CoM, T_bias):
    """
    传统物理补偿模型：
        F_pred = R * G + F_bias
        T_pred = -skew(R * G) * CoM + T_bias
    """
    F_pred = R_mat @ G + F_bias

    V = R_mat @ G
    T_pred = -skew_symmetric(V) @ CoM + T_bias

    return np.concatenate([F_pred, T_pred])


def build_feature(row_dict, G, use_joint_pos=True, use_joint_vel=True):
    quat = np.array([
        float(row_dict['qx']),
        float(row_dict['qy']),
        float(row_dict['qz']),
        float(row_dict['qw'])
    ], dtype=np.float64)

    quat_norm = np.linalg.norm(quat)

    if quat_norm < 1e-12:
        raise RuntimeError("Invalid quaternion with near-zero norm.")

    quat = quat / quat_norm

    # 避免 q 和 -q 跳变
    if quat[3] < 0:
        quat = -quat

    R_mat = R.from_quat(quat).as_matrix()

    # 重力方向在传感器坐标系下的单位向量
    g_sensor = R_mat @ G
    g_sensor = g_sensor / (np.linalg.norm(g_sensor) + 1e-9)

    features = []

    # 3维：重力方向
    features.extend(g_sensor.tolist())

    # 4维：四元数
    features.extend(quat.tolist())

    # 12维：关节角 sin/cos
    if use_joint_pos:
        q = np.array([
            float(row_dict[f'joint_{i}_pos']) for i in range(1, 7)
        ], dtype=np.float64)

        features.extend(np.sin(q).tolist())
        features.extend(np.cos(q).tolist())

    # 6维：关节速度
    if use_joint_vel:
        qd = np.array([
            float(row_dict[f'joint_{i}_vel']) for i in range(1, 7)
        ], dtype=np.float64)

        features.extend(qd.tolist())

    return np.array(features, dtype=np.float32), R_mat


def rmse(x):
    return np.sqrt(np.mean(x ** 2, axis=0))


def plot_compensation_result(
    time_axis,
    traditional_external,
    nn_external,
    output_prefix='compensation_result'
):
    """
    traditional_external:
        传统补偿后的残差，wrench_meas - wrench_physics

    nn_external:
        网络补偿后的残差，wrench_meas - wrench_physics - wrench_residual_nn
    """

    labels_force = ['Fx', 'Fy', 'Fz']
    labels_torque = ['Tx', 'Ty', 'Tz']

    # ---------------------------------------------------------
    # 1. 力补偿效果曲线
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(time_axis, traditional_external[:, i], label='Traditional compensation')
        plt.plot(time_axis, nn_external[:, i], label='Residual network compensation')
        plt.ylabel(f'{labels_force[i]} / N')
        plt.grid(True)
        plt.legend()

    plt.xlabel('Time / s')
    plt.suptitle('Force Compensation Comparison')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_force.png', dpi=200)

    # ---------------------------------------------------------
    # 2. 力矩补偿效果曲线
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))

    for i in range(3):
        idx = i + 3
        plt.subplot(3, 1, i + 1)
        plt.plot(time_axis, traditional_external[:, idx], label='Traditional compensation')
        plt.plot(time_axis, nn_external[:, idx], label='Residual network compensation')
        plt.ylabel(f'{labels_torque[i]} / Nm')
        plt.grid(True)
        plt.legend()

    plt.xlabel('Time / s')
    plt.suptitle('Torque Compensation Comparison')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_torque.png', dpi=200)

    # ---------------------------------------------------------
    # 3. RMSE 柱状图
    # ---------------------------------------------------------
    trad_rmse = rmse(traditional_external)
    nn_rmse = rmse(nn_external)

    names = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, trad_rmse, width, label='Traditional compensation')
    plt.bar(x + width / 2, nn_rmse, width, label='Residual network compensation')
    plt.xticks(x, names)
    plt.ylabel('RMSE')
    plt.title('Compensation RMSE Comparison')
    plt.grid(True, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_rmse.png', dpi=200)

    print(f"\nSaved figures:")
    print(f"  {output_prefix}_force.png")
    print(f"  {output_prefix}_torque.png")
    print(f"  {output_prefix}_rmse.png")

    plt.show()


def main():
    input_csv = './datasets/payload_data1.csv'
    output_csv = './datasets/payload_data_compensated.csv'
    model_file = './models/residual_compensator.pt'

    if not os.path.exists(input_csv):
        print(f"Error: could not find {input_csv}")
        return

    if not os.path.exists(model_file):
        print(f"Error: could not find {model_file}")
        return

    # ---------------------------------------------------------
    # 关键修改：
    # PyTorch 2.6+ 默认 weights_only=True，
    # 你的 checkpoint 中保存了 numpy 数组，所以这里需要 weights_only=False。
    #
    # 注意：只对你自己训练生成的可信模型文件这样做。
    # ---------------------------------------------------------
    checkpoint = torch.load(
        model_file,
        map_location='cpu',
        weights_only=False
    )

    input_dim = checkpoint['input_dim']
    hidden_dim = checkpoint['hidden_dim']

    model = ResidualMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=6
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    x_mean = checkpoint['x_mean']
    x_std = checkpoint['x_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']

    G = checkpoint['G']
    F_bias = checkpoint['F_bias']
    CoM = checkpoint['CoM']
    T_bias = checkpoint['T_bias']

    use_joint_pos = checkpoint['use_joint_pos']
    use_joint_vel = checkpoint['use_joint_vel']

    print("\n================ Loaded Residual Compensator ================")
    print(f"Model file:      {model_file}")
    print(f"Input dim:       {input_dim}")
    print(f"Use joint pos:   {use_joint_pos}")
    print(f"Use joint vel:   {use_joint_vel}")
    print(f"G:               {G}")
    print(f"F_bias:          {F_bias}")
    print(f"CoM:             {CoM}")
    print(f"T_bias:          {T_bias}")
    print("============================================================\n")

    traditional_external_list = []
    nn_external_list = []
    residual_nn_list = []
    physics_list = []
    measured_list = []
    time_list = []

    with open(input_csv, mode='r') as fin, open(output_csv, mode='w', newline='') as fout:
        reader = csv.DictReader(fin)

        if reader.fieldnames is None:
            print("Error: empty CSV.")
            return

        required_cols = ['qx', 'qy', 'qz', 'qw', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']

        for col in required_cols:
            if col not in reader.fieldnames:
                print(f"Error: CSV missing required column: {col}")
                return

        if use_joint_pos:
            for i in range(1, 7):
                col = f'joint_{i}_pos'
                if col not in reader.fieldnames:
                    print(f"Error: model requires joint position, but CSV missing column: {col}")
                    return

        if use_joint_vel:
            for i in range(1, 7):
                col = f'joint_{i}_vel'
                if col not in reader.fieldnames:
                    print(f"Error: model requires joint velocity, but CSV missing column: {col}")
                    return

        extra_cols = [
            'fx_physics', 'fy_physics', 'fz_physics',
            'tx_physics', 'ty_physics', 'tz_physics',

            'fx_traditional_external', 'fy_traditional_external', 'fz_traditional_external',
            'tx_traditional_external', 'ty_traditional_external', 'tz_traditional_external',

            'fx_residual_nn', 'fy_residual_nn', 'fz_residual_nn',
            'tx_residual_nn', 'ty_residual_nn', 'tz_residual_nn',

            'fx_nn_external', 'fy_nn_external', 'fz_nn_external',
            'tx_nn_external', 'ty_nn_external', 'tz_nn_external'
        ]

        fieldnames = reader.fieldnames + extra_cols
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        row_index = 0

        for row in reader:
            feature, R_mat = build_feature(
                row,
                G,
                use_joint_pos=use_joint_pos,
                use_joint_vel=use_joint_vel
            )

            x_norm = (feature - x_mean) / x_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                residual_norm = model(x_tensor).cpu().numpy()[0]

            residual_nn = residual_norm * y_std + y_mean

            wrench_meas = np.array([
                float(row['fx']),
                float(row['fy']),
                float(row['fz']),
                float(row['tx']),
                float(row['ty']),
                float(row['tz'])
            ], dtype=np.float64)

            wrench_physics = physics_predict_wrench(
                R_mat,
                G,
                F_bias,
                CoM,
                T_bias
            )

            # 传统补偿结果：
            # 原始传感器 - 物理模型
            traditional_external = wrench_meas - wrench_physics

            # 网络补偿结果：
            # 原始传感器 - 物理模型 - 网络残差
            nn_external = wrench_meas - wrench_physics - residual_nn

            measured_list.append(wrench_meas)
            physics_list.append(wrench_physics)
            residual_nn_list.append(residual_nn)
            traditional_external_list.append(traditional_external)
            nn_external_list.append(nn_external)

            if 'time_sec' in row:
                time_list.append(float(row['time_sec']))
            else:
                time_list.append(float(row_index))

            values = {}

            values['fx_physics'] = wrench_physics[0]
            values['fy_physics'] = wrench_physics[1]
            values['fz_physics'] = wrench_physics[2]
            values['tx_physics'] = wrench_physics[3]
            values['ty_physics'] = wrench_physics[4]
            values['tz_physics'] = wrench_physics[5]

            values['fx_traditional_external'] = traditional_external[0]
            values['fy_traditional_external'] = traditional_external[1]
            values['fz_traditional_external'] = traditional_external[2]
            values['tx_traditional_external'] = traditional_external[3]
            values['ty_traditional_external'] = traditional_external[4]
            values['tz_traditional_external'] = traditional_external[5]

            values['fx_residual_nn'] = residual_nn[0]
            values['fy_residual_nn'] = residual_nn[1]
            values['fz_residual_nn'] = residual_nn[2]
            values['tx_residual_nn'] = residual_nn[3]
            values['ty_residual_nn'] = residual_nn[4]
            values['tz_residual_nn'] = residual_nn[5]

            values['fx_nn_external'] = nn_external[0]
            values['fy_nn_external'] = nn_external[1]
            values['fz_nn_external'] = nn_external[2]
            values['tx_nn_external'] = nn_external[3]
            values['ty_nn_external'] = nn_external[4]
            values['tz_nn_external'] = nn_external[5]

            for key, value in values.items():
                row[key] = value

            writer.writerow(row)

            row_index += 1

    traditional_external_array = np.vstack(traditional_external_list)
    nn_external_array = np.vstack(nn_external_list)

    time_axis = np.array(time_list, dtype=np.float64)

    # 让时间从 0 开始，曲线更直观
    if len(time_axis) > 0:
        time_axis = time_axis - time_axis[0]

    traditional_rmse = rmse(traditional_external_array)
    nn_rmse = rmse(nn_external_array)

    print(f"Saved compensated CSV to: {output_csv}")

    print("\n================ Compensation RMSE Comparison ================")
    print("Traditional compensation:")
    print(f"  Fx,Fy,Fz RMSE: {traditional_rmse[0]:.4f}, {traditional_rmse[1]:.4f}, {traditional_rmse[2]:.4f} N")
    print(f"  Tx,Ty,Tz RMSE: {traditional_rmse[3]:.4f}, {traditional_rmse[4]:.4f}, {traditional_rmse[5]:.4f} Nm")

    print("\nResidual network compensation:")
    print(f"  Fx,Fy,Fz RMSE: {nn_rmse[0]:.4f}, {nn_rmse[1]:.4f}, {nn_rmse[2]:.4f} N")
    print(f"  Tx,Ty,Tz RMSE: {nn_rmse[3]:.4f}, {nn_rmse[4]:.4f}, {nn_rmse[5]:.4f} Nm")
    print("==============================================================\n")

    plot_compensation_result(
        time_axis=time_axis,
        traditional_external=traditional_external_array,
        nn_external=nn_external_array,
        output_prefix='./imgs/compensation_result'
    )


if __name__ == '__main__':
    main()
