import numpy as np
from matplotlib import pyplot as plt
import time

# 각도 래핑
def angle_wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# 가속도 중력 보정 목적
def Rzyx(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

# Moving Average Filter -> IMU 진동 억제
def smooth(x, N=10):
    return np.convolve(x, np.ones(N)/N, mode='same')

# 상태 정의
numstates=5

# 중력 가속도
g = 9.80665

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

# GPS->Approx Meters
RadiusEarth = 6378388.0 # m
arc= 2.0*np.pi*(RadiusEarth+altitude)/360.0 # m/°

dx = arc * np.cos(latitude*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
dy = arc * np.hstack((0.0, np.diff(latitude))) # in m

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx**2+dy**2)

GPS=(ds!=0.0).astype('bool') # GPS Trigger for Kalman Filter

# sample time (수정 예정!!-현재는 예시 데이터 기반으로 테스트 중)
dt = 1.0/50.0 # Sample Rate of the Measurements is 50Hz
dtGPS=1.0/10.0 # Sample Rate of GPS is 10Hz

# Initial State 구성 (x, y, vx, vy, yaw)
v = speed[0] / 3.6                        # speed [km/h] → [m/s]
psi = np.deg2rad(yaw[0])               # yaw [deg] → [rad]
vx = v * np.cos(psi)                      
vy = v * np.sin(psi)                   

x = np.array([[mx[0]],
               [my[0]],
               [vx],
               [vy],
               [psi]])

# 초기 Uncertainty 행렬 정의
P=np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])

# Process Noise Covariance 행렬 정의
sGPS     = 0.5*8.8*dt**2  # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sCourse  = 0.1*dt # assume 0.1rad/s as maximum turn rate for the vehicle
sVelocity= 8.8*dt # assume 8.8m/s2 as maximum acceleration, forcing the vehicle
sYaw     = 1.0*dt # assume 1.0rad/s2 as the maximum turn rate acceleration for the vehicle
sAccel   = 0.5
Q = np.diag([sGPS**2, sGPS**2, sVelocity**2, sVelocity**2, sYaw**2])

# IMU 입력 노이즈 표준 편차->IMU 오차 공분산
sigma_ax       = 3.0          # m/s^2 - origin: 0.8
sigma_ay       = 3.0          # m/s^2
sigma_yawrate  = np.deg2rad(1.0)  # rad/s - 0.5

Sigma_u = np.diag([sigma_ax**2, sigma_ay**2, sigma_yawrate**2])

# 측정 노이즈 공분산 행렬
varGPS = 1.0 # Standard Deviation of GPS Measurement - 0.1
varyaw = 0.025 # Variance of the yawrate measurement
# _varyaw = 
stdyaw = np.deg2rad(varyaw)
R = np.diag([varGPS**2, varGPS**2, stdyaw**2])

# 계측값 정의
yaw_rad = np.deg2rad(yaw)      # [rad]
yaw_unw = np.unwrap(yaw_rad)   # 경계(±π) 해제
measurements = np.vstack((mx, my, yaw_unw))
# measurements = np.vstack((mx, my, np.deg2rad(yaw)))
m = measurements.shape[1]
# print(m)

# 상태/측정 차원
nx, nz = 5, 3
I = np.eye(nx)

# 저장용
X_est, Y_est, Vx_est, Vy_est, Yaw_est = [], [], [], [], []

# 가속도 이동평균 필터 적용
# ax = smooth(ax, N=10)
# ay = smooth(ay, N=10)

# Extened Kalman Filter
for k in range(m):

    # 입력값 업데이트
    u = np.array([[ax[k]],
                   [ay[k]],
                   [np.deg2rad(yawrate[k])]])

    # 1. A,B 행렬 구하기
    '''
    A, B 행렬은 비선형 -> 선형 근사 행렬이다.
    '''
    psi = float(x[4,0])
    c, s = np.cos(psi), np.sin(psi)
    # ax_k, ay_k = float(u[0,0]), float(u[1,0])

    # 가속도에 roll,pitch에 따른 중력 영향 제거
    acc_body = np.vstack([ax[k], ay[k], az[k]]).reshape(3,1)
    Rbn = Rzyx(np.deg2rad(roll[k]), np.deg2rad(pitch[k]), x[4].item())
    acc_nav = Rbn @ acc_body - np.array([[0],[0],[g]])
    ax_k, ay_k = acc_nav[0].item(), acc_nav[1].item()

    A = np.array([
        [1.0, 0.0, dt,  0.0,  -0.5*(-ax_k*s + ay_k*c)*dt**2],
        [0.0, 1.0, 0.0, dt,   0.5*( ax_k*c - ay_k*s)*dt**2],
        [0.0, 0.0, 1.0, 0.0,        (-ax_k*s - ay_k*c)*dt   ],
        [0.0, 0.0, 0.0, 1.0,         ( ax_k*c - ay_k*s)*dt   ],
        [0.0, 0.0, 0.0, 0.0,  1.0]
    ])

    B = np.array([
        [ 0.5*c*dt**2, -0.5*s*dt**2, 0.0],
        [ 0.5*s*dt**2,  0.5*c*dt**2, 0.0],
        [      c*dt,         -s*dt,  0.0],
        [      s*dt,          c*dt,  0.0],
        [      0.0,           0.0,    dt]
    ])

    # Prediction
    # x=A@x+B@u
    x[0]=x[0]+x[2]*dt+0.5*(c*ax_k-s*ay_k)*dt**2
    x[1]=x[1]+x[3]*dt+0.5*(s*ax_k+c*ay_k)*dt**2
    x[2]=x[2]+(c*ax_k-s*ay_k)*dt
    x[3]=x[3]+(s*ax_k+c*ay_k)*dt
    x[4]=x[4]+u[2]*dt
    # x[4]=angle_wrap(x[4])

    # Project the error covariance ahead
    # 공분산 예측
    P = A @ P @ A.T + B @ Sigma_u @ B.T + Q   

    # Measurement
    z = measurements[:, k].reshape(nz, 1)

    # 계측값 행렬 -> GPS Sensor Rate 다르므로
    if GPS[k]:
        H=np.array([[1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])
    else:
        H=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])        

    # Innovation
    y = z - (H @ x)
    if H[2, 4] == 1.0:            # 현재 스텝에서 yaw를 실제로 업데이트하는 경우에만
        y[2, 0] = angle_wrap(y[2, 0])

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)    

    # Update
    x = x + K @ y
    P = (I - K @ H) @ P

    # 상태 벡터 로그
    X_est.append(x[0].item()); Y_est.append(x[1].item())
    Vx_est.append(x[2].item()); Vy_est.append(x[3].item())
    Yaw_est.append(x[4].item())

# PLOT
fig, axs = plt.subplots(5, 2, figsize=(18, 22))   # 6행 2열 구조
t = np.arange(len(measurements[0])) * dt

# 왼쪽 열 (0열): 기존 그래프들 -----------------------------
axs[0,0].plot(t, mx, label='Measured X (GPS)', alpha=0.6)
axs[0,0].plot(t, X_est, label='Estimated X (EKF)', linewidth=2)
axs[0,0].set_ylabel('X Position [m]')
axs[0,0].legend(loc='best')
axs[0,0].grid(True)

axs[1,0].plot(t, my, label='Measured Y (GPS)', alpha=0.6)
axs[1,0].plot(t, Y_est, label='Estimated Y (EKF)', linewidth=2)
axs[1,0].set_ylabel('Y Position [m]')
axs[1,0].legend(loc='best')
axs[1,0].grid(True)

V_est = np.sqrt(np.array(Vx_est)**2 + np.array(Vy_est)**2)
axs[2,0].plot(t, speed/3.6, label='Measured Speed (GPS)', alpha=0.6)
axs[2,0].plot(t, V_est, label='Estimated Speed (EKF)', linewidth=2)
axs[2,0].set_ylabel('Velocity [m/s]')
axs[2,0].legend(loc='best')
axs[2,0].grid(True)

axs[3,0].plot(t, yaw_unw, label='Measured Yaw (IMU)', alpha=0.6)
axs[3,0].plot(t, Yaw_est, label='Estimated Yaw (EKF)', linewidth=2)
axs[3,0].set_ylabel('Yaw [rad]')
axs[3,0].legend(loc='best')
axs[3,0].grid(True)

a_norm = np.sqrt(ax**2 + ay**2)
axs[4,0].plot(t, ax,    label='ax (IMU)', alpha=0.8)
axs[4,0].plot(t, ay,    label='ay (IMU)', alpha=0.8)
axs[4,0].plot(t, a_norm, label='|a| = sqrt(ax^2+ay^2)', linewidth=2)
axs[4,0].set_xlabel('Time [s]')
axs[4,0].set_ylabel('Acceleration [m/s²]')
axs[4,0].legend(loc='best')
axs[4,0].grid(True)

# 오른쪽 열 (1열): Roll, Pitch -----------------------------
axs[0,1].plot(t, np.deg2rad(roll), label='Roll [rad]', color='orange')
axs[0,1].set_title('Roll Angle (IMU)')
axs[0,1].set_ylabel('Roll [rad]')
axs[0,1].legend(loc='best')
axs[0,1].grid(True)

axs[1,1].plot(t, np.deg2rad(pitch), label='Pitch [rad]', color='green')
axs[1,1].set_title('Pitch Angle (IMU)')
axs[1,1].set_ylabel('Pitch [rad]')
axs[1,1].legend(loc='best')
axs[1,1].grid(True)

# 나머지 오른쪽 칸 비우기 (공백으로 깔끔히)
for i in range(2, 5):
    axs[i,1].axis('off')

plt.tight_layout()
plt.show()

