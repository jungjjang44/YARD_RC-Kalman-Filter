#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

dt = 1.0/50.0
dtGPS=1.0/10.0 # Sample Rate of GPS is 10Hz

# ---------- 프로세스/측정 표준편차 ----------
# (Std 로 두고 제곱해서 분산으로 사용)
px_proc_std = 0.5      # m        (위치 전이 노이즈)
py_proc_std = 0.5      # m
vx_proc_std = 1.0      # m/s      (속도 랜덤워크)
vy_proc_std = 1.0      # m/s
r_proc_std  = 0.02     # rad/s    (yaw rate 랜덤워크)
bg_proc_std = 1e-3     # rad/s    (자이로 바이어스 랜덤워크)

x_meas_std  = 3.0      # m        (위치 측정 표준편차; VIVE면 더 작게)
y_meas_std  = 3.0      # m
gyro_std    = 0.1      # rad/s    (자이로 측정 표준편차)

Q = np.diag([px_proc_std**2, py_proc_std**2,
             vx_proc_std**2, vy_proc_std**2,
             r_proc_std**2,  bg_proc_std**2])

R = np.diag([x_meas_std**2, y_meas_std**2, gyro_std**2])

# ---------- 데이터 로드 ----------
datafile = '2014-03-26-000-Data.csv'
(date, time, millis, ax, ay, az, rollrate, pitchrate, yawrate,
 roll, pitch, yaw, speed, course, latitude, longitude, altitude,
 pdop, hdop, vdop, epe, fix, satellites_view, satellites_used, temp
) = np.loadtxt(datafile, delimiter=',', unpack=True, skiprows=1)

# ---------- 위/경도 -> 평면(m) ----------
RadiusEarth = 6378388.0
arc = 2.0*np.pi*(RadiusEarth+altitude)/360.0
dx = arc * np.cos(latitude*np.pi/180.0) * np.hstack((0.0, np.diff(longitude)))
dy = arc * np.hstack((0.0, np.diff(latitude)))
mx = np.cumsum(dx)
my = np.cumsum(dy)

# ---------- 상태/측정 차원 ----------
nx, nz = 6, 3   # [x,y,vx,vy,r,bg] / [x_meas, y_meas, gyro]
I = np.eye(nx)

# ---------- 초기 상태 ----------
# 속도 초기치는 위치 미분으로 대략 잡아도 됨(여기선 0으로 시작)
x = np.array([[mx[0]],
              [my[0]],
              [0.0],
              [0.0],
              [np.deg2rad(yawrate[0])],
              [0.0]])

P = np.diag([10.0, 10.0, 25.0, 25.0, 1.0, 1.0])

# ---------- 상태전이/측정 행렬 ----------
F = np.array([[1.0, 0.0, dt,  0.0, 0.0, 0.0],  # x_{k+1} = x_k + vx_k dt
              [0.0, 1.0, 0.0, dt,  0.0, 0.0],  # y_{k+1} = y_k + vy_k dt
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # vx 랜덤워크
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # vy 랜덤워크
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # r  랜덤워크
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]) # bg 랜덤워크

# 측정: z = [x_meas, y_meas, gyro] = [x, y, r + bg]
H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

# ---------- 계측 벡터 ----------
z_pos = np.vstack((mx, my))
z_gyro = np.deg2rad(yawrate)
measurements = np.vstack((z_pos, z_gyro))  # shape (3, N)
m = measurements.shape[1]

# ---------- 로그 ----------
X_est, Y_est, VX_est, VY_est, R_est, BG_est = [], [], [], [], [], []

for k in range(m):
    # --- Prediction
    x = F @ x
    P = F @ P @ F.T + Q

    # --- Measurement update (x, y, gyro)
    z = measurements[:, k].reshape(nz, 1)
    yv = z - (H @ x)
    S  = H @ P @ H.T + R
    K  = P @ H.T @ np.linalg.inv(S)
    x  = x + K @ yv
    P  = (I - K @ H) @ P

    # log
    X_est.append(float(x[0])); Y_est.append(float(x[1]))
    VX_est.append(float(x[2])); VY_est.append(float(x[3]))
    R_est.append(float(x[4]));  BG_est.append(float(x[5]))

# ---------- 속도 크기(스칼라) ----------
V_est = np.sqrt(np.array(VX_est)**2 + np.array(VY_est)**2)

# ---------- Plot ----------
fig = plt.figure(figsize=(16,14))

plt.subplot(411)
plt.plot(X_est, Y_est, label='KF position'); plt.scatter(mx[::5], my[::5], s=8, alpha=0.4, label='meas (pos)')
plt.axis('equal'); plt.legend(); plt.title('Position (XY)')

plt.subplot(412)
plt.step(range(m), V_est, label='|v| (KF)')
plt.ylabel('Speed [m/s]'); plt.legend(loc='best')

plt.subplot(413)
plt.step(range(m), R_est, label='r (KF)')
plt.step(range(m), np.deg2rad(yawrate), label='gyro', alpha=0.6)
plt.ylabel('Yaw Rate [rad/s]'); plt.legend(loc='best')

plt.subplot(414)
plt.plot(BG_est, label='b_g (estimated)')
plt.ylabel('Gyro bias [rad/s]'); plt.legend(loc='best'); plt.xlabel('Step')
plt.show()
