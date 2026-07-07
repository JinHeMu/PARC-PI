#!/usr/bin/env python3
"""
方法 3：残差神经网络 —— 动态版

物理重力模型打底，MLP 学习物理解释不了的残差（在动态轨迹下主要就是惯性项）：
    residual      = wrench_meas - wrench_physics
    residual_pred = MLP(features)         # features 含 qdd / qd^2
补偿后残差 = wrench_meas - wrench_physics - residual_pred
"""
import numpy as np
import torch

from common import (
    set_seed, load_csv, compute_kinematic_derivatives,
    identify_physics_params, physics_predict_all,
    build_feature_matrix, feature_dim,
    normalize_fit, train_network, predict_network, rmse_np,
    DEFAULT_FEATURE_CFG,
)


def main():
    filename = './datasets/data2.csv'
    model_filename = './models/residual_compensator.pt'

    set_seed(42)
    cfg = dict(DEFAULT_FEATURE_CFG)

    rows, has_pos, has_vel = load_csv(filename)
    compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3)
    print(f"Loaded {len(rows)} rows | joint_pos={has_pos} joint_vel={has_vel}")
    print(f"Feature config: {cfg} -> dim={feature_dim(cfg)}")

    params = identify_physics_params(rows)
    print("\n================ PHYSICS MODEL ================")
    print(f"Payload Mass (kg): {params['mass']:.4f}")
    print(f"G:      {params['G']}")
    print(f"CoM:    {params['CoM']}")
    print(f"F_bias: {params['F_bias']}")
    print(f"T_bias: {params['T_bias']}")
    print("==============================================")

    X = build_feature_matrix(rows, params, cfg)
    measured = np.vstack([r['wrench'] for r in rows]).astype(np.float32)
    physics = physics_predict_all(rows, params).astype(np.float32)
    Y = measured - physics  # 目标是残差

    before = rmse_np(Y)
    print("\nResidual before NN compensation:")
    print(f"  Fx,Fy,Fz: {before[0]:.4f}, {before[1]:.4f}, {before[2]:.4f} N")
    print(f"  Tx,Ty,Tz: {before[3]:.4f}, {before[4]:.4f}, {before[5]:.4f} Nm")

    x_mean, x_std = normalize_fit(X)
    y_mean, y_std = normalize_fit(Y)
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    input_dim = X.shape[1]
    model = train_network(Xn, Yn, input_dim, hidden_dim=64, epochs=2500, tag='residual-NN')

    pred_residual = predict_network(model, Xn, y_mean, y_std)
    after = rmse_np(Y - pred_residual)
    print("\nResidual after NN compensation:")
    print(f"  Fx,Fy,Fz: {after[0]:.4f}, {after[1]:.4f}, {after[2]:.4f} N")
    print(f"  Tx,Ty,Tz: {after[3]:.4f}, {after[4]:.4f}, {after[5]:.4f} Nm")

    torch.save({
        'method': 'residual_nn',
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim, 'hidden_dim': 64,
        'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std,
        'G': params['G'], 'F_bias': params['F_bias'],
        'CoM': params['CoM'], 'T_bias': params['T_bias'],
        'feature_cfg': cfg,
    }, model_filename)
    print(f"\nSaved residual compensator to: {model_filename}")


if __name__ == '__main__':
    main()
