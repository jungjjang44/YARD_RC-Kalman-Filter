# 필터 사용 안 했을 경우 속도 추정?
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumtrapz

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

# ------------ 유틸 ------------
def Rzyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),  np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def moving_average(x, N=5):
    if N <= 1: return x
    k = np.ones(N)/N
    y = np.convolve(x, k, mode='same')
    # 가장자리 왜곡 완화(선택)
    y[:N//2] = x[:N//2]
    y[-N//2:] = x[-N//2:]
    return y

# ------------ 데이터 로드 (네 코드와 동일 가정) ------------
# date, time, millis, ax, ay, az, rollrate, pitchrate, yawrate, roll, pitch, yaw, speed, course, latitude, longitude, altitude, pdop, hdop, vdop, epe, fix, satellites_view, satellites_used, temp
# = np.loadtxt(...)

# ------------ 시간/각도 준비 ------------
g = 9.80665
t = (millis - millis[0]) / 1000.0
if np.any(np.diff(t) <= 0):
    dt = 1.0/50.0
    t = np.arange(len(yaw)) * dt
else:
    # 샘플 간격 평균치
    dt = np.mean(np.diff(t))

yaw_unw = np.unwrap(np.deg2rad(yaw))
roll_r  = np.deg2rad(roll)
pitch_r = np.deg2rad(pitch)

# ------------ body->nav 회전 + 중력 보정 ------------
N = len(t)
ax_nav = np.zeros(N); ay_nav = np.zeros(N); az_nav = np.zeros(N)

for k in range(N):
    Rbn = Rzyx(roll_r[k], pitch_r[k], yaw_unw[k])
    acc_body = np.array([[ax[k]],[ay[k]],[az[k]]])  # m/s^2
    acc_nav  = Rbn @ acc_body - np.array([[0.0],[0.0],[g]])
    ax_nav[k], ay_nav[k], az_nav[k] = acc_nav[:,0]

# (선택) 저역통과(이동평균)로 고주파 노이즈 완화
ax_nav_f = moving_average(ax_nav, N=5)
ay_nav_f = moving_average(ay_nav, N=5)

# ------------ 정지(near-static) 구간에서 바이어스 추정/제거 ------------
v_gps = speed / 3.6
yawrate_meas = np.deg2rad(yawrate)

static_mask = (v_gps < 0.3) & (np.abs(yawrate_meas) < np.deg2rad(0.5)) \
              & (np.hypot(ax_nav_f, ay_nav_f) < 0.2)

if np.any(static_mask):
    bx = np.median(ax_nav_f[static_mask])
    by = np.median(ay_nav_f[static_mask])
else:
    bx = by = 0.0  # 정지 구간 없으면 0으로
ax_nav_c = ax_nav_f - bx
ay_nav_c = ay_nav_f - by

# ------------ 적분으로 속도 추정 (벡터 → 스칼라) ------------
# 초기 속도 벡터는 GPS 속도+yaw로 설정
v0  = v_gps[0]
psi = yaw_unw
vx0 = v0 * np.cos(psi[0])
vy0 = v0 * np.sin(psi[0])

# 누적 적분: v(t) = v0 + ∫ a dt (trapezoidal)
vx_est = vx0 + cumtrapz(ax_nav_c, t, initial=0.0)
vy_est = vy0 + cumtrapz(ay_nav_c, t, initial=0.0)

v_est  = np.hypot(vx_est, vy_est)   # 추정 스칼라 속도
v_gpss = v_gps                      # 측정 스칼라 속도

# ------------ 성능 지표(간단) ------------
rmse = np.sqrt(np.mean((v_est - v_gpss)**2))
bias = np.mean(v_est - v_gpss)

print(f"Speed RMSE (est vs GPS): {rmse:.3f} m/s,  Bias: {bias:.3f} m/s")
print(f"Estimated accel bias removed: bx={bx:.3f}, by={by:.3f} m/s^2")

# ------------ 플롯 ------------
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# (1) 가속도 원시 vs 보정
axs[0].plot(t, ax_nav,  label='ax_nav (raw)',  alpha=0.6)
axs[0].plot(t, ay_nav,  label='ay_nav (raw)',  alpha=0.6)
axs[0].plot(t, ax_nav_c, label='ax_nav (comp+debias)', linewidth=2)
axs[0].plot(t, ay_nav_c, label='ay_nav (comp+debias)', linewidth=2)
axs[0].set_ylabel('a_nav [m/s²]')
axs[0].legend(loc='best'); axs[0].grid(True)

# (2) 속도 비교 (스칼라)
axs[1].plot(t, v_gpss, label='Speed (GPS)', alpha=0.8)
axs[1].plot(t, v_est,  label='Speed (Integrated IMU)', linewidth=2)
axs[1].set_ylabel('Speed [m/s]')
axs[1].legend(loc='best'); axs[1].grid(True)

# (3) 속도 벡터 성분 비교(참고)
axs[2].plot(t, vx_est, label='vx_est')
axs[2].plot(t, vy_est, label='vy_est')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('v-components [m/s]')
axs[2].legend(loc='best'); axs[2].grid(True)

plt.tight_layout()
plt.show()
