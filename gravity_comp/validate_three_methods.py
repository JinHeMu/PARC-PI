#!/usr/bin/env python3
"""
validate_three_methods.py — 用独立 val 数据集验证三种已训练好的方法

与 compare_three_methods.py 的区别：
    compare : 在一份数据上现场训练三种方法再对比（用于开发调参）
    validate: 【加载已保存的 checkpoint】，在【单独的 val 数据集】上评估泛化能力
              （不重新训练，不在 val 上重算归一化，正确的泛化验证）

三种方法：
    traditional : pred = physics                       （物理参数来自 checkpoint）
    plain_nn    : pred = MLP_plain(feat)               （直接回归 wrench）
    residual_nn : pred = physics + MLP_res(feat)       （物理 + 残差）
补偿后外力 external = measured - pred，无接触时越接近 0 越好。

用法：
    python validate_three_methods.py \
        --val ./datasets/val.csv \
        --residual-model ./models/residual_compensator.pt \
        --plain-model ./models/traditional_net.pt \
        --out ./imgs/val
"""
import os
import csv
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import (
    load_csv, compute_kinematic_derivatives, get_time_axis,
    physics_predict_all, build_feature_matrix,
    WrenchMLP, rmse_np, CHANNELS, UNITS, DEFAULT_FEATURE_CFG,
)

COLORS = {'trad': '#d62728', 'plain': '#1f77b4', 'res': '#2ca02c'}
LABELS = {'trad': 'Traditional (physics LSQ)',
          'plain': 'Plain NN',
          'res': 'Residual NN (physics + NN)'}


# ============================================================
# checkpoint 工具
# ============================================================
def load_ckpt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型文件: {path}")
    return torch.load(path, map_location='cpu', weights_only=False)


def params_from_ckpt(ckpt):
    """从 checkpoint 取物理参数，供物理预测与特征构造用。"""
    return {
        'G': np.asarray(ckpt['G'], dtype=np.float64),
        'F_bias': np.asarray(ckpt['F_bias'], dtype=np.float64),
        'CoM': np.asarray(ckpt['CoM'], dtype=np.float64),
        'T_bias': np.asarray(ckpt['T_bias'], dtype=np.float64),
    }


def nn_predict(ckpt, rows):
    """
    用 checkpoint 里的模型对 rows 做前向。
    特征用 checkpoint 的 feature_cfg 构造，并用【训练时】的 x_mean/x_std 归一化，
    输出再用【训练时】的 y_mean/y_std 反归一化。
    """
    params = params_from_ckpt(ckpt)
    cfg = ckpt.get('feature_cfg', DEFAULT_FEATURE_CFG)

    X = build_feature_matrix(rows, params, cfg)
    x_mean = np.asarray(ckpt['x_mean'], dtype=np.float32)
    x_std = np.asarray(ckpt['x_std'], dtype=np.float32)
    y_mean = np.asarray(ckpt['y_mean'], dtype=np.float32)
    y_std = np.asarray(ckpt['y_std'], dtype=np.float32)

    Xn = (X - x_mean) / x_std

    model = WrenchMLP(ckpt['input_dim'], ckpt['hidden_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
    return out * y_std + y_mean


# ============================================================
# 指标
# ============================================================
def metrics(external):
    """返回 (rmse, mae, maxabs)，均为 (6,)。"""
    return (np.sqrt(np.mean(external ** 2, axis=0)),
            np.mean(np.abs(external), axis=0),
            np.max(np.abs(external), axis=0))


def print_table(name, ext):
    rmse, mae, mx = metrics(ext)
    print(f"\n----- {name} -----")
    print(f"{'ch':>4} | {'RMSE':>8} | {'MAE':>8} | {'MaxAbs':>8}  ({'unit'})")
    for i, ch in enumerate(CHANNELS):
        print(f"{ch:>4} | {rmse[i]:>8.4f} | {mae[i]:>8.4f} | {mx[i]:>8.4f}  ({UNITS[i]})")
    print(f"  Force RMSE mean: {rmse[:3].mean():.4f} N | "
          f"Torque RMSE mean: {rmse[3:].mean():.4f} Nm")


# ============================================================
# 出图
# ============================================================
def plot_timeseries(t, results, window, out):
    w0, w1 = window
    w1 = min(w1, len(t)) if w1 > 0 else len(t)
    tw = t[w0:w1]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    order = [0, 3, 1, 4, 2, 5]
    for k, ch in enumerate(order):
        ax = axes[k // 2, k % 2]
        ax.axhline(0, color='k', lw=0.6, alpha=0.4)
        for key in ('trad', 'plain', 'res'):
            ax.plot(tw, results[key][w0:w1, ch], color=COLORS[key],
                    lw=1.0, alpha=0.85, label=LABELS[key])
        ax.set_ylabel(f'{CHANNELS[ch]} residual / {UNITS[ch]}')
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc='upper right', fontsize=9)
    axes[2, 0].set_xlabel('Time / s'); axes[2, 1].set_xlabel('Time / s')
    fig.suptitle('Validation set — Compensated External Wrench Residual (closer to 0 is better)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=200); plt.close(fig)


def plot_rmse(results, out):
    Rt = rmse_np(results['trad'])
    Rp = rmse_np(results['plain'])
    Rr = rmse_np(results['res'])
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
    ax.set_title('Validation RMSE across 6 DOF', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def save_metrics_csv(results, out):
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'channel', 'unit', 'RMSE', 'MAE', 'MaxAbs'])
        for key, name in [('trad', 'traditional'), ('plain', 'plain_nn'), ('res', 'residual_nn')]:
            rmse, mae, mx = metrics(results[key])
            for i, ch in enumerate(CHANNELS):
                w.writerow([name, ch, UNITS[i],
                            f'{rmse[i]:.6f}', f'{mae[i]:.6f}', f'{mx[i]:.6f}'])


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default='./datasets/val.csv', help='验证集 CSV（独立于训练集）')
    ap.add_argument('--residual-model', default='./models/residual_compensator.pt')
    ap.add_argument('--plain-model', default='./models/traditional_net.pt')
    ap.add_argument('--physics-from', default='residual', choices=['residual', 'plain'],
                    help='传统方法的物理参数取自哪个 checkpoint（两者应一致）')
    ap.add_argument('--out', default='./imgs/val', help='输出前缀')
    ap.add_argument('--window', type=int, nargs=2, default=[0, 0],
                    help='时间序列绘图窗口 [起, 止]（样本索引），默认全画')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # ---- 加载 val 数据并算加速度 ----
    rows, has_pos, has_vel = load_csv(args.val)
    compute_kinematic_derivatives(rows, smooth=True, window=11, poly=3)
    N = len(rows)
    t = get_time_axis(rows)
    print(f"Validation set: {args.val} | {N} rows | joint_pos={has_pos} joint_vel={has_vel}")

    measured = np.vstack([r['wrench'] for r in rows]).astype(np.float32)

    # ---- 加载 checkpoint ----
    res_ckpt = load_ckpt(args.residual_model)
    plain_ckpt = load_ckpt(args.plain_model)

    # 一致性检查：两个 checkpoint 的物理参数应当一致（同一训练集辨识）
    g_diff = np.linalg.norm(np.asarray(res_ckpt['G']) - np.asarray(plain_ckpt['G']))
    if g_diff > 1e-6:
        print(f"[warn] 两个 checkpoint 的 G 不一致 (‖ΔG‖={g_diff:.3e})，"
              f"可能不是同一训练集训练的。")

    phys_ckpt = res_ckpt if args.physics_from == 'residual' else plain_ckpt
    params = params_from_ckpt(phys_ckpt)
    print(f"Physics params from: {args.physics_from} checkpoint | "
          f"mass≈{np.linalg.norm(params['G'])/9.81:.4f} kg")

    # ---- 三种方法预测 ----
    physics = physics_predict_all(rows, params).astype(np.float32)

    ext_trad = measured - physics                                   # 方法1
    pred_plain = nn_predict(plain_ckpt, rows).astype(np.float32)
    ext_plain = measured - pred_plain                               # 方法2
    res_pred = nn_predict(res_ckpt, rows).astype(np.float32)
    ext_res = measured - physics - res_pred                         # 方法3

    results = {'trad': ext_trad, 'plain': ext_plain, 'res': ext_res}

    # ---- 打印指标 ----
    print("\n================ VALIDATION METRICS ================")
    print_table('Traditional', ext_trad)
    print_table('Plain NN', ext_plain)
    print_table('Residual NN', ext_res)

    print("\n================ RMSE 对比表 (N / Nm) ================")
    Rt, Rp, Rr = rmse_np(ext_trad), rmse_np(ext_plain), rmse_np(ext_res)
    print(f"{'Channel':>8} | {'Traditional':>12} | {'Plain NN':>10} | {'Residual NN':>12} | {'Res vs Trad':>11}")
    print("-" * 66)
    for i, ch in enumerate(CHANNELS):
        imp = (1 - Rr[i] / Rt[i]) * 100 if Rt[i] > 1e-9 else 0.0
        print(f"{ch:>8} | {Rt[i]:>12.4f} | {Rp[i]:>10.4f} | {Rr[i]:>12.4f} | {imp:>10.1f}%")
    print("=" * 66)

    # ---- 出图 + 存指标 ----
    plot_timeseries(t, results, tuple(args.window), f'{args.out}_timeseries.png')
    plot_rmse(results, f'{args.out}_rmse.png')
    save_metrics_csv(results, f'{args.out}_metrics.csv')

    print(f"\nSaved: {args.out}_timeseries.png")
    print(f"       {args.out}_rmse.png")
    print(f"       {args.out}_metrics.csv")


if __name__ == '__main__':
    main()

