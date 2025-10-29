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
# plt.figure(figsize=(16,6))
# plt.plot(t, yawrate_est,  label='yaw rate est (d/dt of yaw)', linewidth=2)
# plt.plot(t, yawrate_meas, label='yaw rate mea (IMU)', alpha=0.8)
# plt.xlabel('time [s]')
# plt.ylabel('yaw rate [rad/s]')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(16,6))
# plt.plot(t, yaw,  label='yaw rate est (d/dt of yaw)', linewidth=2)
# # plt.plot(t, yawrate_meas, label='yaw rate mea (IMU)', alpha=0.8)
# plt.xlabel('time [s]')
# plt.ylabel('yaw rate [rad/s]')
# plt.legend(loc='best')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # PLOT
# yaw=np.deg2rad(yaw)
# fig = plt.figure(figsize=(16,16))
# plt.step(range(len(yaw)),yaw, label='$\dot \psi$')
# plt.ylabel('yaw')
# plt.ylim([-5,5])
# plt.legend(loc='best')
# plt.show()
# =========================
# 중력/경사 보정 가속도 비교 플롯
# =========================

# 1) 회전행렬 (Z-Y-X, yaw-pitch-roll) 정의: body -> navigation
def Rzyx(roll, pitch, yaw):
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx   # body->nav

# 2) 라디안 변환 (yaw는 언랩 사용)
roll_rad  = np.deg2rad(roll)
pitch_rad = np.deg2rad(pitch)
yaw_rad_u = yaw_unw.copy()      # 이미 위에서 계산한 언랩 yaw [rad]

# 3) 중력 보정 적용
g = 9.80665
N = len(t)
ax_nav = np.zeros(N)
ay_nav = np.zeros(N)
az_nav = np.zeros(N)

for k in range(N):
    Rbn = Rzyx(roll_rad[k], pitch_rad[k], yaw_rad_u[k])
    acc_body = np.array([[ax[k]], [ay[k]], [az[k]]])     # [m/s^2]
    acc_nav  = Rbn @ acc_body - np.array([[0.0],[0.0],[g]])  # 중력 제거
    ax_nav[k] = acc_nav[0,0]
    ay_nav[k] = acc_nav[1,0]
    az_nav[k] = acc_nav[2,0]

# 4) 크기 비교용
a_norm_raw = np.sqrt(ax**2 + ay**2)
a_norm_nav = np.sqrt(ax_nav**2 + ay_nav**2)

# 5) 플롯 (원시 vs 보정)
fig2, axs2 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

axs2[0].plot(t, ax,     label='ax (raw, body)', alpha=0.7)
axs2[0].plot(t, ax_nav, label='ax (nav, gravity-comp)', linewidth=2)
axs2[0].set_ylabel('ax [m/s²]')
axs2[0].legend(loc='best'); axs2[0].grid(True)

axs2[1].plot(t, ay,     label='ay (raw, body)', alpha=0.7)
axs2[1].plot(t, ay_nav, label='ay (nav, gravity-comp)', linewidth=2)
axs2[1].set_ylabel('ay [m/s²]')
axs2[1].legend(loc='best'); axs2[1].grid(True)

axs2[2].plot(t, a_norm_raw, label='|a| (raw, body)', alpha=0.7)
axs2[2].plot(t, a_norm_nav, label='|a| (nav, gravity-comp)', linewidth=2)
axs2[2].set_xlabel('Time [s]')
axs2[2].set_ylabel('|a| [m/s²]')
axs2[2].legend(loc='best'); axs2[2].grid(True)

plt.tight_layout()
plt.show()

# 1) yaw 언랩 + body->nav + 중력 보정
yaw_unw = np.unwrap(np.deg2rad(yaw))
roll_r, pitch_r, yaw_r = np.deg2rad(roll), np.deg2rad(pitch), yaw_unw
g = 9.80665
ax_nav, ay_nav = np.zeros_like(ax), np.zeros_like(ay)
for k in range(len(yaw)):
    Rbn = Rzyx(roll_r[k], pitch_r[k], yaw_r[k])
    acc_body = np.array([[ax[k]],[ay[k]],[az[k]]])
    acc_nav  = Rbn @ acc_body - np.array([[0],[0],[g]])
    ax_nav[k], ay_nav[k] = acc_nav[0,0], acc_nav[1,0]

yawrate_meas = np.deg2rad(yawrate)   # [rad/s]

# 2) 정지(near-static) 마스크
v_gps = speed/3.6
mask = (v_gps < 0.3) & (np.abs(yawrate_meas) < np.deg2rad(0.5)) \
       & (np.hypot(ax_nav, ay_nav) < 0.2)

# 3) 바이어스/공분산 추정
Z = np.vstack([ax_nav[mask], ay_nav[mask], yawrate_meas[mask]]).T  # [N x 3]
bias = Z.mean(axis=0)                      # [b_ax, b_ay, b_ψ̇]
Sigma_u = np.cov((Z - bias), rowvar=False) # 3x3 공분산 (입력 노이즈)
print("bias:", bias)
print("Sigma_u:\n", Sigma_u)
