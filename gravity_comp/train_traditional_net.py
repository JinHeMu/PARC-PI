#!/usr/bin/env python3
"""
方法 2：传统（纯）神经网络 —— 动态版

直接用 MLP 把 (姿态, 关节角, 关节速度, 关节加速度, 向心项) 映射到 6 维 wrench：
    wrench_pred = MLP(features)
补偿后残差 = wrench_meas - wrench_pred

动态要点：读入后先 compute_kinematic_derivatives() 算出平滑速度和加速度，
再构造含 qdd / qd^2 的特征。
"""
import numpy as np
import torch

from common import (
    set_seed, load_csv, compute_kinematic_derivatives,
    identify_physics_params, build_feature_matrix, feature_dim,
    normalize_fit, train_network, predict_network, rmse_np,
    DEFAULT_FEATURE_CFG,
)


def main():
    filename = './datasets/data2.csv'
    model_filename = './models/traditional_net.pt'

    set_seed(42)
    cfg = dict(DEFAULT_FEATURE_CFG)  # 动态轨迹：pos+vel+acc+vel^2 全开

    rows, has_pos, has_vel = load_csv(filename)
    compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3)
    print(f"Loaded {len(rows)} rows | joint_pos={has_pos} joint_vel={has_vel}")
    print(f"Feature config: {cfg} -> dim={feature_dim(cfg)}")

    # 复用同一套特征需要 G，因此仍辨识一次物理参数（方法2 本身不用物理预测）
    params = identify_physics_params(rows)

    X = build_feature_matrix(rows, params, cfg)
    Y = np.vstack([r['wrench'] for r in rows]).astype(np.float32)  # 直接回归整段 wrench

    x_mean, x_std = normalize_fit(X)
    y_mean, y_std = normalize_fit(Y)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    input_dim = X.shape[1]
    model = train_network(Xn, Yn, input_dim, hidden_dim=64, epochs=2500, tag='plain-NN')

    pred = predict_network(model, Xn, y_mean, y_std)
    err = rmse_np(Y - pred)
    print("\nCompensation RMSE (plain NN):")
    print(f"  Fx,Fy,Fz: {err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f} N")
    print(f"  Tx,Ty,Tz: {err[3]:.4f}, {err[4]:.4f}, {err[5]:.4f} Nm")

    torch.save({
        'method': 'plain_nn',
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim, 'hidden_dim': 64,
        'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std,
        'G': params['G'], 'F_bias': params['F_bias'],
        'CoM': params['CoM'], 'T_bias': params['T_bias'],
        'feature_cfg': cfg,
    }, model_filename)
    print(f"\nSaved plain NN model to: {model_filename}")


if __name__ == '__main__':
    main()
