HARDCODING=False

from math import cos, sin
import json
from datetime import datetime
import numpy as np
from collections import deque  # (남김: 필요 시 로그 길이 제한용)
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from custom_msgs.msg import Localization
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

# ====== PARAMETER ======
PS = 0.1           # 초기 공분산 대각 성분
NX, NZ = 5, 3
I = np.eye(NX)

# Process Noise Covariance (모델 불확실성)
sTracker  = 0.003 # 0.003
sVelocity = 0.003
sYaw      = 0.003

_imuF = 8.0
# IMU 입력 노이즈 표준편차 -> 입력 공분산
sigma_ax      = 0.11 * _imuF             # m/s^2
sigma_ay      = 0.11 * _imuF               # m/s^2
sigma_yawrate = np.deg2rad(0.1)      # rad/s

# 측정 노이즈 공분산 행렬(Tracker/GNSS 등)
varGPS = 0.003                        # 1σ [m]
varyaw = 0.0025                      # yaw 측정(라디안 변환 전) 분산 성분의 "deg" 기준 값

class EKF(Node):
    def __init__(self):
        super().__init__('ekf')

        # 상태: [x, y, vx, vy, yaw]^T
        self.states = np.zeros((NX, 1))

        # 공분산, 잡음 행렬
        self.P = np.diag([PS, PS, PS, PS, PS])
        self.Q = np.diag([sTracker**2, sTracker**2, sVelocity**2, sVelocity**2, sYaw**2])
        self.Sigma_u = np.diag([sigma_ax**2, sigma_ay**2, sigma_yawrate**2])

        # H, R
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # x
            [0.0, 1.0, 0.0, 0.0, 0.0],  # y
            [0.0, 0.0, 0.0, 0.0, 1.0]   # yaw
        ])
        self.R = np.diag([varGPS**2, varGPS**2, np.deg2rad(varyaw)**2])

        # 타이밍
        self.t_last = None
        self.default_dt_clip = (1e-3, 0.2)

        # 현재 측정(Tracker)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.pose_ts = 0.0

        # 플래그
        self.init = True
        self.pose_trigger = False
        self.imu_trigger = False

        # IMU 누적 버퍼(평균용)
        self.reset_accumulators()

        # 퍼블리셔
        self.pub_v = self.create_publisher(Float64MultiArray, '/filtered_v', 10)
        self.pub_a = self.create_publisher(Float64MultiArray, '/filtered_a', 10)
        self.pub_raw_a = self.create_publisher(Float64MultiArray, '/raw_a', 10)
        
        # 구독
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(Localization, '/localization_info', self.pose_callback, 10)

        # 로그(필요 최소)
        self.X_est, self.Y_est, self.Vx_est, self.Vy_est, self.Yaw_est, self.time = [], [], [], [], [], []
        self.X_meas, self.Y_meas, self.Yaw_meas = [], [], []
        self.ax_used, self.ay_used, self.wz_used = [], [], []

    # ---------- Utils ----------
    def reset_accumulators(self):
        self.acc_ax = 0.0
        self.acc_ay = 0.0
        self.acc_az = 0.0     # <--- 추가: 수평 투영에 필요
        self.roll = 0.0
        self.pitch = 0.0
        self.acc_wz = 0.0     # rad/s
        self.acc_count = 0

    def save_estimates_to_json(self, filepath: str = None):
        if filepath is None:
            filepath = "src/extended_kalman_filter/data/ekf_estimates.json"
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "count": len(self.X_est)
            },
            "estimates": {
                "Time":      list(self.time),
                "x_est":     list(self.X_est),
                "y_est":     list(self.Y_est),
                "vx_est":    list(self.Vx_est),
                "vy_est":    list(self.Vy_est),
                "yaw_est":   list(self.Yaw_est),
                "x_meas":    list(self.X_meas),
                "y_meas":    list(self.Y_meas),
                "yaw_meas":  list(self.Yaw_meas),
                "ax_used":   list(self.ax_used),
                "ay_used":   list(self.ay_used),
                "wz_used":   list(self.wz_used)
            }
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.get_logger().info(f"EKF 추정치 JSON 저장 완료: {filepath}")

    # ---------- Callbacks ----------
    def imu_callback(self, msg: Imu):
        # 타임스탬프(미사용), 자세 쿼터니언 -> 롤/피치(rad)
        q = [msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w]
        r = R.from_quat(q)
        roll, pitch, _ = r.as_euler('xyz', degrees=False)

        # 가속도(m/s^2), 자이로 z(deg/s 가정 → rad/s 변환)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        wz = np.deg2rad(msg.angular_velocity.z)

        # 포즈 콜백 사이 구간 동안 단순 평균을 위한 누적
        self.acc_ax += ax
        self.acc_ay += ay
        self.acc_az += az          # <--- 추가
        self.acc_wz += wz
        self.roll   += roll
        self.pitch  += pitch
        # self.acc_ax = ax
        # self.acc_ay = ay
        # self.acc_az = az          # <--- 추가
        # self.acc_wz = wz
        # self.roll   = roll
        # self.pitch  = pitch
        self.acc_count += 1
        self.imu_trigger = True

    def pose_callback(self, msg: Localization):
        # 필수 측정 수신
        self.pose_trigger = True
        self.pose_ts = msg.timestamp.sec + msg.timestamp.nanosec * 1e-9

        # 위치/자세(라디안) 읽기
        px = msg.px
        py = msg.py
        yaw = np.deg2rad(msg.p_yaw)

        # 센서 오프셋 보정(Body->Nav), 필요시 유지
        offset = -0.07  # [m]
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.x = px + cy * offset
        self.y = py + sy * offset
        self.yaw = yaw

        # 초기화
        if self.init and self.imu_trigger:
            self.states = np.array([[self.x],
                                    [self.y],
                                    [0.0],
                                    [0.0],
                                    [self.yaw]])
            self.init = False
            self.t_last = self.pose_ts  # 첫 포즈 시간 설정
            # 초기 한 번은 업데이트만 수행하고 반환(예측에 필요한 dt가 없음)
            self.measurement_update()
            self.finalize_step_and_publish(ax_used=0.0, ay_used=0.0, wz_used=0.0)
            self.reset_accumulators()
            return

        if not (self.pose_trigger and self.imu_trigger):
            return

        # dt 계산 및 안정화
        if self.t_last is None:
            self.t_last = self.pose_ts
            return
        dt = float(np.clip(self.pose_ts - self.t_last, *self.default_dt_clip))
        self.get_logger().info(
            f"dt = {dt})"
        )

        # 누적 평균
        if self.acc_count > 0:
            ax_k    = self.acc_ax / self.acc_count
            ay_k    = self.acc_ay / self.acc_count
            az_k    = self.acc_az / self.acc_count
            wz_k    = self.acc_wz / self.acc_count
            roll_k  = self.roll   / self.acc_count
            pitch_k = self.pitch  / self.acc_count
            # ax_k    = self.acc_ax
            # ay_k    = self.acc_ay
            # az_k    = self.acc_az
            # wz_k    = self.acc_wz
            # roll_k  = self.roll  
            # pitch_k = self.pitch 

        else:
            ax_k = ay_k = az_k = wz_k = roll_k = pitch_k = 0.0

        _a_msg = Float64MultiArray()
        _a_msg.data=[ay_k,-ax_k]
        self.pub_raw_a.publish(_a_msg)

        # 가속도 중력 보상 및 좌표 변환
        ###### 여기 부분에 넣어야 함!!
        # self.get_logger().info(
        #     f"[PoseCB] roll_avg = {roll_k:.4f} rad ({np.rad2deg(roll_k):.2f}°), "
        #     f"pitch_avg = {pitch_k:.4f} rad ({np.rad2deg(pitch_k):.2f}°)"
        # )

        # (1) 중력 방향 단위벡터(g_hat) 계산: 롤/피치 기반
        gx = -np.sin(pitch_k)
        gy =  np.sin(roll_k) * np.cos(pitch_k)
        gz =  np.cos(roll_k) * np.cos(pitch_k)
        g_hat = np.array([gx, gy, gz])
        g_hat = g_hat / (np.linalg.norm(g_hat) + 1e-12)

        # (2) 가속도 벡터에서 중력 성분 제거(수평 평면 투영)
        a_vec  = np.array([ax_k, ay_k, az_k])      # m/s^2
        a_proj = a_vec - np.dot(a_vec, g_hat) * g_hat
        ax_corr, ay_corr, _ = a_proj

        # (3) IMU가 시계방향 90° 회전 보정: ax=ay, ay=-ax
        ax_use = ay_corr
        ay_use = -ax_corr
        ######

        if HARDCODING:
            tmp_ax=ax_k-(-0.38612)
            tmp_ay=ay_k-(-0.15855)

            # TMP
            ax_use = tmp_ay
            ay_use = -tmp_ax
        

        # ========= EKF Prediction =========
        x = self.states
        psi = x[4, 0]
        c, s = np.cos(psi), np.sin(psi)

        # 선형화 행렬 A, 입력 행렬 B  (보정된 가속도 사용!)
        A = np.array([
            [1.0, 0.0, dt,  0.0,  -0.5*(-ax_use*s + ay_use*c)*dt**2],
            [0.0, 1.0, 0.0, dt,   0.5*( ax_use*c - ay_use*s)*dt**2],
            [0.0, 0.0, 1.0, 0.0,        (-ax_use*s - ay_use*c)*dt   ],
            [0.0, 0.0, 0.0, 1.0,         ( ax_use*c - ay_use*s)*dt   ],
            [0.0, 0.0, 0.0, 0.0,  1.0]
        ])
        B = np.array([
            [ 0.5*c*dt**2, -0.5*s*dt**2, 0.0],
            [ 0.5*s*dt**2,  0.5*c*dt**2, 0.0],
            [      c*dt,         -s*dt,  0.0],
            [      s*dt,          c*dt,  0.0],
            [      0.0,           0.0,    dt]
        ])

        # 상태 예측(비선형 상태식 직접 적용) — 보정된 가속도 사용
        x_pred = np.empty_like(x)
        x_pred[0,0] = x[0,0] + x[2,0]*dt + 0.5*(c*ax_use - s*ay_use)*dt**2
        x_pred[1,0] = x[1,0] + x[3,0]*dt + 0.5*(s*ax_use + c*ay_use)*dt**2
        x_pred[2,0] = x[2,0] + (c*ax_use - s*ay_use)*dt
        x_pred[3,0] = x[3,0] + (s*ax_use + c*ay_use)*dt
        x_pred[4,0] = x[4,0] + wz_k*dt
        self.states = x_pred

        # 공분산 예측
        self.P = A @ self.P @ A.T + B @ self.Sigma_u @ B.T + self.Q

        # ========= EKF Measurement Update =========
        self.measurement_update()

        # ========= Step 마무리 =========
        self.t_last = self.pose_ts
        self.finalize_step_and_publish(ax_used=ax_use, ay_used=ay_use, wz_used=wz_k)

        # 포즈 하나 처리 후 IMU 누적 리셋
        self.reset_accumulators()

    # ---------- EKF 서브루틴 ----------
    def measurement_update(self):
        z = np.array([[self.x], [self.y], [self.yaw]])
        y = z - (self.H @ self.states)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.states = self.states + K @ y
        self.P = (I - K @ self.H) @ self.P

    def finalize_step_and_publish(self, ax_used: float, ay_used: float, wz_used: float):
        # 로그
        self.X_est.append(self.states[0,0]); self.Y_est.append(self.states[1,0])
        self.Vx_est.append(self.states[2,0]); self.Vy_est.append(self.states[3,0])
        self.Yaw_est.append(self.states[4,0]); self.time.append(self.pose_ts)
        self.X_meas.append(self.x); self.Y_meas.append(self.y); self.Yaw_meas.append(self.yaw)
        self.ax_used.append(ax_used); self.ay_used.append(ay_used); self.wz_used.append(wz_used)

        # 퍼블리시: 속도
        v_msg = Float64MultiArray()
        v_msg.data = [self.states[2,0], self.states[3,0]]
        self.pub_v.publish(v_msg)

        # 퍼블리시: 평균 가속도(디버그/모니터링용) — 보정 후 값
        a_msg = Float64MultiArray()
        a_msg.data = [ax_used, ay_used]
        self.pub_a.publish(a_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKF()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"{e}")
        raise
    finally:
        node.save_estimates_to_json()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
