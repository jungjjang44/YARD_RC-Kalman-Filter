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

x = np.matrix([[mx[0]],
               [my[0]],
               [vx],
               [vy],
               [psi]])


