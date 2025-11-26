#!/usr/bin/env python
"""快速测试局地温度方程闭合率 - 单时间步"""
import numpy as np
import h5py as h5

ALPHA, BETA, GAMMA = 0.01, 0.01, 0.1

print("Loading data...")
with h5.File('/gz-data/ERA5_2023_weekly_new.h5', 'r') as f:
    surf0, surf1 = f['surface'][0], f['surface'][1]
    upper0, upper1 = f['upper_air'][0], f['upper_air'][1]

dt = 7 * 24 * 3600
R_d, c_p, g, L_v = 287.0, 1004.0, 9.8, 2.5e6
plevs = np.array([200, 300, 500, 700, 850]) * 100

# 提取变量
t0, t1 = upper0[2], upper1[2]  # 温度
u, v, w = upper1[3], upper1[4], upper1[5]  # 风场
q, o3, clwc, ciwc = upper1[6], upper1[0], upper1[9], upper1[8]

# 1. 观测温度趋势
dT_dt_obs = (t1 - t0) / dt

# 2. 水平平流
lat = np.linspace(-90, 90, 721) * np.pi / 180
cos_lat = np.cos(lat).reshape(1, -1, 1)
dx = 6.371e6 * 0.25 * np.pi / 180

t_px = np.pad(t1, ((0,0), (0,0), (1,1)), mode='wrap')
t_py = np.pad(t1, ((0,0), (1,1), (0,0)), mode='edge')
dT_dx = (t_px[:,:,2:] - t_px[:,:,:-2]) / (2 * dx * cos_lat)
dT_dy = (t_py[:,2:,:] - t_py[:,:-2,:]) / (2 * dx)
h_adv = -(u * dT_dx + v * dT_dy)

# 3. 垂直运动
p3d = plevs.reshape(5, 1, 1)
dT_dp = np.gradient(t1, plevs, axis=0)
v_adv = -((R_d * t1 / (c_p * p3d) - dT_dp) * w)

# 4. 非绝热加热 (简化)
A_sw = surf1[0] - surf1[15]  # TOA - surface SW
A_lw = surf1[1] - surf1[16]  # TOA - surface LW
w_rad = 0.5*q + 0.3*o3 + 0.2*(clwc+ciwc)
w_rad = w_rad / w_rad.sum(axis=0, keepdims=True).clip(min=1e-10)
Q_rad = (g/c_p) * (A_sw + A_lw) * w_rad / 2500 / 1000

precip = surf1[4] + surf1[5]
lat_prof = np.array([0.1, 0.2, 0.4, 0.2, 0.1]).reshape(5,1,1)
Q_lat = (L_v/c_p) * precip * lat_prof / 1000

Q_total = Q_rad + Q_lat

# 5. 理论温度趋势
dT_dt_theo = ALPHA * h_adv + BETA * v_adv + GAMMA * Q_total
residual = dT_dt_obs - dT_dt_theo

# 统计
print(f"\n{'='*60}")
print(f"温度局地变化方程测试 (ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA})")
print(f"{'='*60}")
print(f"\n观测 |∂T/∂t| 均值: {np.mean(np.abs(dT_dt_obs)):.2e} K/s")
print(f"理论 |∂T/∂t| 均值: {np.mean(np.abs(dT_dt_theo)):.2e} K/s")
print(f"残差 |res| 均值:   {np.mean(np.abs(residual)):.2e} K/s")
print(f"RMSE: {np.sqrt(np.mean(residual**2)):.2e} K/s")

closure = 1 - np.mean(np.abs(residual)) / np.mean(np.abs(dT_dt_obs))
print(f"\n总体闭合率: {closure*100:.1f}%")

print(f"\n各层闭合率:")
for i, p in enumerate([200, 300, 500, 700, 850]):
    c = 1 - np.mean(np.abs(residual[i])) / np.mean(np.abs(dT_dt_obs[i]))
    print(f"  {p} hPa: {c*100:.1f}%")
