#!/usr/bin/env python3
"""
compare_three_methods.py — 一键对比三种补偿方法并出图（动态版）

    方法 1 传统方法      : 物理最小二乘（重力模型，动态下不含惯性项，作为基线）
    方法 2 传统神经网络   : MLP 直接回归 wrench，输入含 qdd/qd^2
    方法 3 残差神经网络   : 物理模型 + MLP 补残差，输入含 qdd/qd^2

动态要点：读入后立刻 compute_kinematic_derivatives() 算平滑速度与加速度
（在打乱之前，保证时间连续），再统一 80/20 随机划分做公平对比。

产物：
    imgs/fig1_residual_timeseries_6dof.png
    imgs/fig2_rmse_bars.png
    imgs/fig3_rmse_all_6dof.png
    datasets/compensated_three_methods.csv
"""
import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')  # 无界面也能出图；需窗口显示可注释
import matplotlib.pyplot as plt

from common import (
    set_seed, load_csv, compute_kinematic_derivatives, get_time_axis,
    identify_physics_params, physics_predict_all,
    build_feature_matrix, feature_dim, normalize_fit,
    train_network, predict_network, rmse_np,
    CHANNELS, UNITS, DEFAULT_FEATURE_CFG,
)

COLORS = {'trad': '#d62728', 'plain': '#1f77b4', 'res': '#2ca02c'}
LABELS = {'trad': 'Traditional (physics LSQ)',
          'plain': 'Plain NN',
          'res': 'Residual NN (physics + NN)'}


def run_all_methods(rows, params, cfg, train_mask, epochs=2500, seed=42):
    measured = np.vstack([r['wrench'] for r in rows]).astype(np.float32)
    physics = physics_predict_all(rows, params).astype(np.float32)

    X = build_feature_matrix(rows, params, cfg)
    x_mean, x_std = normalize_fit(X[train_mask])  # 用训练集统计量，防泄露
    Xn = (X - x_mean) / x_std
    input_dim = X.shape[1]

    # 方法1 传统
    ext_trad = measured - physics

    # 方法2 传统神经网络
    y2m, y2s = normalize_fit(measured[train_mask])
    Yn2 = (measured - y2m) / y2s
    net2 = train_network(Xn[train_mask], Yn2[train_mask], input_dim,
                         epochs=epochs, seed=seed, tag='plain-NN')
    pred2 = predict_network(net2, Xn, y2m, y2s)
    ext_plain = measured - pred2

    # 方法3 残差神经网络
    Y3 = measured - physics
    y3m, y3s = normalize_fit(Y3[train_mask])
    Yn3 = (Y3 - y3m) / y3s
    net3 = train_network(Xn[train_mask], Yn3[train_mask], input_dim,
                         epochs=epochs, seed=seed, tag='residual-NN')
    pred3 = predict_network(net3, Xn, y3m, y3s)
    ext_res = measured - physics - pred3

    return {'measured': measured, 'physics': physics,
            'trad': ext_trad, 'plain': ext_plain, 'res': ext_res}


def plot_timeseries(t, result, window=(200, 700), out='imgs/fig1_residual_timeseries_6dof.png'):
    w0, w1 = window
    w1 = min(w1, len(t)); tw = t[w0:w1]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    order = [0, 3, 1, 4, 2, 5]
    for k, ch in enumerate(order):
        ax = axes[k // 2, k % 2]
        ax.axhline(0, color='k', lw=0.6, alpha=0.4)
        ax.plot(tw, result['trad'][w0:w1, ch], color=COLORS['trad'], lw=1.0, alpha=0.85, label=LABELS['trad'])
        ax.plot(tw, result['plain'][w0:w1, ch], color=COLORS['plain'], lw=1.0, alpha=0.85, label=LABELS['plain'])
        ax.plot(tw, result['res'][w0:w1, ch], color=COLORS['res'], lw=1.2, alpha=0.9, label=LABELS['res'])
        ax.set_ylabel(f'{CHANNELS[ch]} residual / {UNITS[ch]}')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc='upper right', fontsize=9)
    axes[2, 0].set_xlabel('Time / s'); axes[2, 1].set_xlabel('Time / s')
    fig.suptitle('Compensated External Wrench Residual — 6 DOF (closer to 0 is better)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=200); plt.close(fig)


def plot_rmse_bars(result, test_idx, out='imgs/fig2_rmse_bars.png'):
    rt = rmse_np(result['trad'][test_idx])
    rp = rmse_np(result['plain'][test_idx])
    rr = rmse_np(result['res'][test_idx])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(3); wd = 0.25
    for ax, sl, title, unit in [(a1, slice(0, 3), 'Force RMSE', 'N'),
                                (a2, slice(3, 6), 'Torque RMSE', 'Nm')]:
        ax.bar(x - wd, rt[sl], wd, color=COLORS['trad'], label=LABELS['trad'])
        ax.bar(x,      rp[sl], wd, color=COLORS['plain'], label=LABELS['plain'])
        ax.bar(x + wd, rr[sl], wd, color=COLORS['res'], label=LABELS['res'])
        ax.set_xticks(x); ax.set_xticklabels(CHANNELS[sl])
        ax.set_ylabel(f'RMSE / {unit}'); ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle('Test-set Compensation RMSE — Three Methods', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=200); plt.close(fig)
    return rt, rp, rr


def plot_rmse_all(result, out='imgs/fig3_rmse_all_6dof.png'):
    Rt, Rp, Rr = rmse_np(result['trad']), rmse_np(result['plain']), rmse_np(result['res'])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(6); wd = 0.25
    ax.bar(x - wd, Rt, wd, color=COLORS['trad'], label=LABELS['trad'])
    ax.bar(x,      Rp, wd, color=COLORS['plain'], label=LABELS['plain'])
    ax.bar(x + wd, Rr, wd, color=COLORS['res'], label=LABELS['res'])
    for i in range(6):
        for off, v in [(-wd, Rt[i]), (0, Rp[i]), (wd, Rr[i])]:
            ax.text(i + off, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(CHANNELS)
    ax.set_ylabel('RMSE (N or Nm)')
    ax.set_title('Overall Compensation RMSE across 6 DOF', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_compensated_csv(t, result, out='datasets/compensated_three_methods.csv'):
    header = ['time']
    for tag in ['meas', 'trad', 'plain', 'res']:
        header += [f'{c.lower()}_{tag}' for c in CHANNELS]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for i in range(len(t)):
            row = [t[i]]
            row += result['measured'][i].tolist()
            row += result['trad'][i].tolist()
            row += result['plain'][i].tolist()
            row += result['res'][i].tolist()
            w.writerow(row)


def main():
    filename = './datasets/data1.csv'
    os.makedirs('imgs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    set_seed(42)

    cfg = dict(DEFAULT_FEATURE_CFG)  # 动态：pos+vel+acc+vel^2 全开

    rows, has_pos, has_vel = load_csv(filename)
    compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3)
    N = len(rows); t = get_time_axis(rows)
    print(f"Loaded {N} rows | joint_pos={has_pos} joint_vel={has_vel}")
    print(f"Feature config: {cfg} -> dim={feature_dim(cfg)}")

    perm = np.arange(N); np.random.shuffle(perm)
    ntr = int(N * 0.8)
    train_mask = np.zeros(N, dtype=bool); train_mask[perm[:ntr]] = True
    test_idx = perm[ntr:]

    params = identify_physics_params([rows[i] for i in perm[:ntr]])
    print(f"\nPayload mass: {params['mass']:.4f} kg")

    result = run_all_methods(rows, params, cfg, train_mask, epochs=2500, seed=42)

    plot_timeseries(t, result)
    rt, rp, rr = plot_rmse_bars(result, test_idx)
    plot_rmse_all(result)
    save_compensated_csv(t, result)

    print("\n================ Test-set RMSE (N / Nm) ================")
    print(f"{'Channel':>8} | {'Traditional':>12} | {'Plain NN':>10} | {'Residual NN':>12}")
    print("-" * 54)
    for i, ch in enumerate(CHANNELS):
        print(f"{ch:>8} | {rt[i]:>12.4f} | {rp[i]:>10.4f} | {rr[i]:>12.4f}")
    print("=" * 54)
    print("\nSaved: imgs/fig1_residual_timeseries_6dof.png, imgs/fig2_rmse_bars.png, "
          "imgs/fig3_rmse_all_6dof.png, datasets/compensated_three_methods.csv")


if __name__ == '__main__':
    main()
