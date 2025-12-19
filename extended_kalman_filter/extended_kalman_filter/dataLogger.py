from math import *
import json
from datetime import datetime
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from custom_msgs.msg import Localization
from sensor_msgs.msg import Imu

class DataLogger(Node):
    def __init__(self):
        super().__init__('imu_loc_json_logger')

        # 로그 버퍼
        self.imu_log = []   # dict 리스트
        self.loc_log = []   # dict 리스트

        # 마지막 수신 시각(옵션)
        self.last_imu_t = None
        self.last_loc_t = None

        # 구독자
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 50
        )
        self.loc_sub = self.create_subscription(
            Localization, '/localization_info', self.loc_callback, 30
        )

        self.get_logger().info('📥 IMU & Localization JSON 로거 시작')

    # IMU 콜백: 선가속도(ax,ay,az m/s^2), 각속도(gx,gy,gz rad/s)
    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        gx = float(msg.angular_velocity.x)  # rad/s (ROS 표준)
        gy = float(msg.angular_velocity.y)
        gz = float(msg.angular_velocity.z)

        self.imu_log.append({
            "t": t,
            "ax_mps2": ax,
            "ay_mps2": ay,
            "az_mps2": az,
            "gx_rads": gx,
            "gy_rads": gy,
            "gz_rads": gz
        })
        self.last_imu_t = t

    # Localization 콜백: 위치(px,py)와 attitude(roll/pitch/yaw)
    def loc_callback(self, msg: Localization):
        t = msg.timestamp.sec + msg.timestamp.nanosec * 1e-9

        px = float(msg.px)
        py = float(msg.py)

        roll_deg  = float(msg.p_roll)
        pitch_deg = float(msg.p_pitch)
        yaw_deg   = float(msg.p_yaw)

        roll_rad  = radians(roll_deg)
        pitch_rad = radians(pitch_deg)
        yaw_rad   = radians(yaw_deg)

        self.loc_log.append({
            "t": t,
            "px_m": px,
            "py_m": py,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "roll_rad": roll_rad,
            "pitch_rad": pitch_rad,
            "yaw_rad": yaw_rad
        })
        self.last_loc_t = t

    def save_to_json(self, filepath: str = None):
        # 파일 경로 자동 생성
        if filepath is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"src/extended_kalman_filter/data/data_log.json"

        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "imu_count": len(self.imu_log),
                "loc_count": len(self.loc_log),
            },
            "imu": self.imu_log,
            "localization": self.loc_log
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f"💾 JSON 저장 완료: {filepath}")

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"{e}")
        raise
    finally:
        node.save_to_json()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
