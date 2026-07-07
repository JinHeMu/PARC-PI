#!/usr/bin/env python3
"""
方法 1：传统最小二乘物理辨识（standalone）

模型：
    F_meas = R * G + F_bias
    T_meas = -skew(R * G) * CoM + T_bias

只依赖 numpy / scipy，可独立运行，不需要 torch。
"""
import numpy as np

from common import load_csv, identify_physics_params, physics_predict_all, rmse_np


def main():
    filename = './datasets/data2.csv'

    rows, has_pos, has_vel = load_csv(filename)
    N = len(rows)
    print(f"Loaded {N} data points from {filename}")

    if N < 6:
        print("Error: need at least 6 distinct data points.")
        return

    print("\n--- Least Squares Identification ---")
    params = identify_physics_params(rows)

    G = params['G']
    CoM = params['CoM']
    F_bias = params['F_bias']
    T_bias = params['T_bias']
    mass = params['mass']

    print("\n================ IDENTIFICATION RESULTS ================")
    print(f"Payload Mass (kg):        {mass:.4f}")
    print(f"Center of Mass [x,y,z]:   [{CoM[0]:.4f}, {CoM[1]:.4f}, {CoM[2]:.4f}] (m)")
    print(f"Gravity Vector G:         [{G[0]:.4f}, {G[1]:.4f}, {G[2]:.4f}] (N)")
    print(f"Force Bias  [x,y,z]:      [{F_bias[0]:.4f}, {F_bias[1]:.4f}, {F_bias[2]:.4f}] (N)")
    print(f"Torque Bias [x,y,z]:      [{T_bias[0]:.4f}, {T_bias[1]:.4f}, {T_bias[2]:.4f}] (Nm)")
    print("========================================================")

    # 补偿后的外力残差（无外部接触时应接近 0）
    measured = np.vstack([r['wrench'] for r in rows])
    physics = physics_predict_all(rows, params)
    external = measured - physics
    err = rmse_np(external)

    print("\nCompensation RMSE (traditional method):")
    print(f"  Fx,Fy,Fz: {err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f} N")
    print(f"  Tx,Ty,Tz: {err[3]:.4f}, {err[4]:.4f}, {err[5]:.4f} Nm")


if __name__ == '__main__':
    main()
