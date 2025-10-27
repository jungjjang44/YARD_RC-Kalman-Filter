#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

# sample time - IMU 값으로 변경
dt = 1.0/50.0

v_proc_std   = 8.8*dt            # m/s (v 전이 노이즈)
r_proc_std   = 1.0*dt            # rad/s (r 전이 노이즈)
a_proc_std   = 0.5               # m/s^2 (a 전이 노이즈)
bg_proc_std  = 1e-3              # rad/s (gyro bias 랜덤워크)
ba_proc_std  = 5e-2              # m/s^2 (acc  bias 랜덤워크)

r_meas_std   = 0.1               # rad/s (자이로 측정 표준편차)
a_meas_std   = 1.0               # m/s^2 (가속도 측정 표준편차)

Q = np.diag([v_proc_std**2, r_proc_std**2, a_proc_std**2,
             bg_proc_std**2, ba_proc_std**2])
R = np.diag([r_meas_std**2, a_meas_std**2])

# 센서 데이터 불러오기
datafile = '2014-03-26-000-Data.csv'
(date, time, millis, ax, ay, az, rollrate, pitchrate, yawrate,
 roll, pitch, yaw, speed, course, latitude, longitude, altitude,
 pdop, hdop, vdop, epe, fix, satellites_view, satellites_used, temp
) = np.loadtxt(datafile, delimiter=',', unpack=True, skiprows=1)

g = 9.81
pitch_rad = np.deg2rad(pitch)
roll_rad  = np.deg2rad(roll)
ax_corr = ax - g*np.sin(pitch_rad) # 중력 보정

# nx: 상태 벡터 / nz: 계측
nx, nz = 5, 2     # [v, r, a, b_g, b_a]  /  [gyro, acc]
I = np.eye(nx)

# 초기 상태 벡터
x = np.array([[speed[0]/3.6 + 1e-3],           # v0
              [yawrate[0]*np.pi/180.0],        # r0
              [ax_corr[0]],                    # a0
              [0.0],                           # b_g0
              [0.0]])                          # b_a0

P = np.diag([100.0, 100.0, 100.0, 1.0, 1.0])   


F = np.array([[1.0, 0.0, dt, 0.0, 0.0],        # v_{k+1} = v_k + a_k*dt
              [0.0, 1.0, 0.0, 0.0, 0.0],       # r  랜덤워크
              [0.0, 0.0, 1.0, 0.0, 0.0],       # a  랜덤워크
              [0.0, 0.0, 0.0, 1.0, 0.0],       # b_g 랜덤워크
              [0.0, 0.0, 0.0, 0.0, 1.0]])      # b_a 랜덤워크


H = np.array([[0.0, 1.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 1.0]])


z_gyro = np.deg2rad(yawrate)    # rad/s
z_acc  = ax_corr                # m/s^2 (중력 보정 반영)
measurements = np.vstack((z_gyro, z_acc))
m = measurements.shape[1]


use_gps_v = True
gps_every = 5                   # 50Hz에서 5스텝마다 10Hz
v_meas_std = 0.8                # m/s (대략적인 표준편차)
H_v = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])  # v 를 직접 관측
R_v = np.array([[v_meas_std**2]])


V_est, R_est, A_est, BG_est, BA_est = [], [], [], [], []

for k in range(m):
    # --- Prediction
    x = F @ x
    P = F @ P @ F.T + Q

    z = measurements[:, k].reshape(nz, 1)
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (I - K @ H) @ P

    if use_gps_v and (k % gps_every == 0):
        z_v = np.array([[speed[k]/3.6]])  # m/s
        yv  = z_v - (H_v @ x)
        Sv  = H_v @ P @ H_v.T + R_v
        Kv  = P @ H_v.T @ np.linalg.inv(Sv)
        x   = x + Kv @ yv
        P   = (I - Kv @ H_v) @ P

    # log
    V_est.append(float(x[0]))
    R_est.append(float(x[1]))
    A_est.append(float(x[2]))
    BG_est.append(float(x[3]))
    BA_est.append(float(x[4]))

fig = plt.figure(figsize=(16,16))

plt.subplot(411)
plt.step(range(m), V_est, label='v (KF)')
plt.step(range(m), speed/3.6, label='v (GPS ref)', alpha=0.6)
plt.ylabel('Velocity [m/s]'); plt.ylim([0,30]); plt.legend(loc='best')

plt.subplot(412)
plt.step(range(m), R_est, label='r (KF)')
plt.step(range(m), np.deg2rad(yawrate), label='r (gyro ref)', alpha=0.6)
plt.ylabel('Yaw Rate [rad/s]'); plt.ylim([-0.6,0.6]); plt.legend(loc='best')

plt.subplot(413)
plt.step(range(m), A_est, label='a (KF)')
plt.step(range(m), ax_corr, label='a (IMU corrected)', alpha=0.6)
plt.ylabel('Accel [m/s^2]'); plt.legend(loc='best')

plt.subplot(414)
plt.plot(BG_est, label='b_g'); plt.plot(BA_est, label='b_a')
plt.ylabel('Bias'); plt.legend(loc='best'); plt.xlabel('Step')
plt.show()
