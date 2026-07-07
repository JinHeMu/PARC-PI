#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
import os

def skew_symmetric(v):
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])

def main():
    filename = './datasets/data2.csv'

    if not os.path.exists(filename):
        print(f"Error: Could not find {filename}")
        return

    recorded_R = []
    recorded_F = []
    recorded_T = []

    print(f"Reading data from {filename}...")
    with open(filename, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            quat = [float(row['qx']), float(row['qy']),
                    float(row['qz']), float(row['qw'])]
            recorded_R.append(R.from_quat(quat).as_matrix())
            recorded_F.append([float(row['fx']), float(row['fy']), float(row['fz'])])
            recorded_T.append([float(row['tx']), float(row['ty']), float(row['tz'])])

    N = len(recorded_R)
    print(f"Successfully loaded {N} valid data points.")

    if N < 6:
        print("Error: Need at least 6 distinct data points for a reliable calculation.")
        return

    print("\n--- Starting Least Squares Calculation ---")

    # ---------------------------------------------------------
    # 1. 辨识重力向量和力偏置
    #    模型：F_meas = R * G + F_bias
    # ---------------------------------------------------------
    A_force = np.zeros((3 * N, 6))
    B_force = np.zeros((3 * N, 1))

    for i in range(N):
        A_force[i*3:(i+1)*3, 0:3] = recorded_R[i]
        A_force[i*3:(i+1)*3, 3:6] = np.eye(3)
        B_force[i*3:(i+1)*3, 0]   = recorded_F[i]

    X_force, residuals_F, rank_F, s_F = np.linalg.lstsq(A_force, B_force, rcond=None)

    G      = X_force[0:3].flatten()
    F_bias = X_force[3:6].flatten()
    mass   = np.linalg.norm(G) / 9.81

    # ---------------------------------------------------------
    # 2. 辨识质心和力矩偏置
    #    模型：T_meas = -skew(R * G) * CoM + T_bias
    # ---------------------------------------------------------
    A_torque = np.zeros((3 * N, 6))
    B_torque = np.zeros((3 * N, 1))

    for i in range(N):
        V       = recorded_R[i] @ G
        V_skew  = skew_symmetric(V)
        A_torque[i*3:(i+1)*3, 0:3] = -V_skew
        A_torque[i*3:(i+1)*3, 3:6] = np.eye(3)
        B_torque[i*3:(i+1)*3, 0]   = recorded_T[i]

    X_torque, residuals_T, rank_T, s_T = np.linalg.lstsq(A_torque, B_torque, rcond=None)

    CoM    = X_torque[0:3].flatten()
    T_bias = X_torque[3:6].flatten()

    # ---------------------------------------------------------
    # 输出结果（已移除底盘倾角估计）
    # ---------------------------------------------------------
    print("\n================ IDENTIFICATION RESULTS ================")
    print(f"Payload Mass (kg):        {mass:.4f}")
    print(f"Center of Mass [x,y,z]:   [{CoM[0]:.4f}, {CoM[1]:.4f}, {CoM[2]:.4f}] (m)")
    print(f"Force Bias [x,y,z]:       [{F_bias[0]:.2f}, {F_bias[1]:.2f}, {F_bias[2]:.2f}] (N)")
    print(f"Torque Bias [x,y,z]:      [{T_bias[0]:.2f}, {T_bias[1]:.2f}, {T_bias[2]:.2f}] (Nm)")
    print("========================================================\n")

    if residuals_F.size > 0:
        print(f"Force Fitting Error  (Sum of Squared Residuals): {residuals_F[0]:.4f}")
    if residuals_T.size > 0:
        print(f"Torque Fitting Error (Sum of Squared Residuals): {residuals_T[0]:.4f}")

if __name__ == '__main__':
    main()
