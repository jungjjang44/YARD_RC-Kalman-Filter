#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

dt = 1.0/50.0
sVelocity= 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw     = 1.0*dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel   = 0.5
varyaw = 0.1 # Variance of the yawrate measurement
varacc = 1.0 # Variance of the longitudinal Acceleration

# Q, R (너가 이미 정의한 값 사용)
Q = np.diag([sVelocity**2, sYaw**2, sAccel**2])
R = np.diag([varyaw, varacc**2])

# 데이터 파일 불러오기
datafile = '2014-03-26-000-Data.csv'
date, time, millis, ax, ay, az, rollrate, pitchrate, yawrate, roll, pitch, yaw, speed, course, latitude, longitude, altitude, pdop, hdop, vdop, epe, fix, satellites_view, satellites_used, temp = np.loadtxt(datafile, delimiter=',', unpack=True, 
                  #converters={1: mdates.strpdate2num('%H%M%S%f'),
                  #            0: mdates.strpdate2num('%y%m%d')},
                  skiprows=1)

# 상태/측정 차원
nx, nz = 3, 2
I = np.eye(nx)

# 초기 상태 (속도[m/s], yawrate[rad/s], ax[m/s^2])
x = np.array([[speed[0]/3.6 + 1e-3],
              [yawrate[0]*np.pi/180.0],
              [ax[0]]])

# 초기 공분산
P = np.diag([100.0, 100.0, 100.0])

# 상태전이/측정 행렬
F = np.array([[1.0, 0.0, dt],
              [0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

H = np.array([[0.0, 1.0, 0.0],
              [0.0, 0.0, 1.0]])

ax=ax-9.81*np.sin(pitch*np.pi/180.0)

# 계측 (yawrate[rad/s], ax[m/s^2])
measurements = np.vstack((yawrate*np.pi/180.0, ax))
m = measurements.shape[1]

# 저장용
V_est, R_est, A_est = [], [], []

for k in range(m):
    # --- Prediction
    x = F @ x
    P = F @ P @ F.T + Q

    # --- Measurement
    z = measurements[:, k].reshape(nz, 1)

    # Innovation
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # Update
    x = x + K @ y
    P = (I - K @ H) @ P

    # log
    V_est.append(float(x[0])); R_est.append(float(x[1])); A_est.append(float(x[2]))



# PLOT
fig = plt.figure(figsize=(16,16))

plt.subplot(311)
plt.step(range(len(measurements[0])),V_est, label='$v$')
plt.step(range(len(measurements[0])),speed/3.6, label='$v$ (from GPS as reference)', alpha=0.6)
plt.ylabel('Velocity')
plt.ylim([0,30])
plt.legend(loc='best')

plt.subplot(312)
plt.step(range(len(measurements[0])),R_est, label='$\dot \psi$')
plt.step(range(len(measurements[0])),yawrate/180.0*np.pi, label='$\dot \psi$ (from IMU as reference)', alpha=0.6)
plt.ylabel('Yaw Rate')
plt.ylim([-0.6, 0.6])
plt.legend(loc='best')

plt.subplot(313)
plt.step(range(len(measurements[0])),A_est, label='$a$')
plt.step(range(len(measurements[0])),ax, label='$a$ (from IMU as reference)', alpha=0.6)
plt.ylabel('Accelation')
plt.legend(loc='best')
plt.xlabel('Filter Step')

plt.show()