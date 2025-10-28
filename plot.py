import numpy as np
from matplotlib import pyplot as plt
from sympy import Matrix
import time

# 상태 정의
numstates=5

# Data Load
datafile = '2014-03-26-000-Data.csv'

date, \
time, \
millis, \
ax, \
ay, \
az, \
rollrate, \
pitchrate, \
yawrate, \
roll, \
pitch, \
yaw, \
speed, \
course, \
latitude, \
longitude, \
altitude, \
pdop, \
hdop, \
vdop, \
epe, \
fix, \
satellites_view, \
satellites_used, \
temp = np.loadtxt(datafile, delimiter=',', unpack=True, 
                  #converters={1: mdates.strpdate2num('%H%M%S%f'),
                  #            0: mdates.strpdate2num('%y%m%d')},
                  skiprows=1)

print('Read \'%s\' successfully.' % datafile)

# --- 시간축 구성: millis가 ms라고 가정 ---
t = (millis - millis[0]) / 1000.0  # [s]
# 시간이 비단조이면(드물지만) 고정 dt로 대체
if np.any(np.diff(t) <= 0):
    dt = 1.0/50.0
    t = np.arange(len(yaw)) * dt

# --- 각도(라디안) 처리 & 래핑 해제 ---
yaw_deg = yaw.copy()                 # 원본 보관
yaw_rad = np.deg2rad(yaw_deg)        # [rad]
yaw_unw = np.unwrap(yaw_rad)         # 래핑(±π) 경계 해제

# --- 수치 미분으로 추정 yaw rate 계산 ---
# 불균등 샘플링도 대응: np.gradient(값, 시간)
yawrate_est = np.gradient(yaw_unw, t)   # [rad/s]

# --- IMU 측정 yawrate를 rad/s로 변환 ---
yawrate_meas = np.deg2rad(yawrate)      # [rad/s]

# (선택) 간단 저역통과로 더 매끈하게 보고 싶다면 주석 해제
# def smooth(x, N=5): return np.convolve(x, np.ones(N)/N, mode='same')
# yawrate_est  = smooth(yawrate_est, 5)
# yawrate_meas = smooth(yawrate_meas, 5)

# --- 플롯 ---
plt.figure(figsize=(16,6))
plt.plot(t, yawrate_est,  label='yaw rate est (d/dt of yaw)', linewidth=2)
plt.plot(t, yawrate_meas, label='yaw rate mea (IMU)', alpha=0.8)
plt.xlabel('time [s]')
plt.ylabel('yaw rate [rad/s]')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# # PLOT
# yaw=np.deg2rad(yaw)
# fig = plt.figure(figsize=(16,16))
# plt.step(range(len(yaw)),yaw, label='$\dot \psi$')
# plt.ylabel('yaw')
# plt.ylim([-5,5])
# plt.legend(loc='best')
# plt.show()