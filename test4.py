import numpy as np
from matplotlib import pyplot as plt
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
sigma_ax       = 0.8          # m/s^2
sigma_ay       = 0.8          # m/s^2
sigma_yawrate  = np.deg2rad(0.5)  # rad/s

Sigma_u = np.diag([sigma_ax**2, sigma_ay**2, sigma_yawrate**2])

# 측정 노이즈 공분산 행렬
varGPS = 5.0 # Standard Deviation of GPS Measurement
varyaw = 0.1 # Variance of the yawrate measurement
R = np.diag([varGPS**2, varGPS**2, varyaw**2])

# 계측값 정의
measurements = np.vstack((mx, my, np.deg2rad(yaw)))
m = measurements.shape[1]
print(m)

# 상태/측정 차원
nx, nz = 5, 3
I = np.eye(nx)

# 상태전이/측정 행렬
# A: 5*5
A=np.array([[1.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])

# 저장용
X_est, Y_est, Vx_est, Vy_est, Yaw_est = [], [], [], [], []

# Extened Kalman Filter
for k in range(m):

    # 입력값 업데이트
    u = np.array([[ax[k]],
                   [ay[k]],
                   [yawrate[k]]])

    # 1. A,B 행렬 구하기
    '''
    A, B 행렬은 비선형 -> 선형 근사 행렬이다.
    '''
    psi = float(x[4,0])
    c, s = np.cos(psi), np.sin(psi)
    ax_k, ay_k = float(u[0,0]), float(u[1,0])

    A = np.array([
        [1.0, 0.0, dt,  0.0,  0.5*(-ax_k*s - ay_k*c)*dt**2],
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
    x=A@x+B@u

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
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)    

    # Update
    x = x + K @ y
    P = (I - K @ H) @ P

    # 상태 벡터 로그
    # 위치
    X_est.append(float(x[0])); Y_est.append(float(x[1]))
    
    # 속도
    Vx_est.append(float(x[2])); Vy_est.append(float(x[3]))
    
    # Yaw
    Yaw_est.append(float(x[4]))

# PLOT
fig = plt.figure(figsize=(16,16))

# plt.subplot(311)
# plt.step(range(len(measurements[0])),V_est, label='$v$')
# plt.step(range(len(measurements[0])),speed/3.6, label='$v$ (from GPS as reference)', alpha=0.6)
# plt.ylabel('Velocity')
# plt.ylim([0,30])
# plt.legend(loc='best')

# plt.subplot(312)
# plt.step(range(len(measurements[0])),R_est, label='$\dot \psi$')
# plt.step(range(len(measurements[0])),yawrate/180.0*np.pi, label='$\dot \psi$ (from IMU as reference)', alpha=0.6)
# plt.ylabel('Yaw Rate')
# plt.ylim([-0.6, 0.6])
# plt.legend(loc='best')

# plt.subplot(313)
# plt.step(range(len(measurements[0])),A_est, label='$a$')
# plt.step(range(len(measurements[0])),ax, label='$a$ (from IMU as reference)', alpha=0.6)
# plt.ylabel('Accelation')
# plt.legend(loc='best')
# plt.xlabel('Filter Step')

# plt.show()
