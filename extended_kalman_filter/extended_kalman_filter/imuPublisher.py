#!/usr/bin/env python3
import threading
import json
import redis
import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import Imu


class RedisPubSubToIMU(Node):
    def __init__(self, host='127.0.0.1', port=6379, channel='imu_data', frame_id='imu_link'):
        super().__init__('redis_pubsub_to_imu')

        # ROS 퍼블리셔
        self.publisher_ = self.create_publisher(Imu, '/imu/data', 10)
        self.frame_id = frame_id

        # Redis 연결
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.pubsub = self.r.pubsub()
        self.pubsub.subscribe(channel)

        self.get_logger().info(f"📡 Redis Pub/Sub 채널 '{channel}' 구독 시작")
        self.get_logger().info(f"🛰️ 퍼블리시 토픽: /imu/data, frame_id={self.frame_id}")

        # 별도 스레드로 수신 루프 실행
        self._thr = threading.Thread(target=self._listen_loop, daemon=True)
        self._thr.start()

    def _listen_loop(self):
        """Redis Pub/Sub 수신 루프 (Blocking)"""
        for message in self.pubsub.listen():
            if not rclpy.ok():
                break
            if message['type'] != 'message':
                continue

            raw = message['data']
            imu_msg = self.parse_to_imu(raw)
            if imu_msg:
                imu_msg.header.stamp = self.get_clock().now().to_msg()
                imu_msg.header.frame_id = self.frame_id
                self.publisher_.publish(imu_msg)
                # self.get_logger().debug(f"📥 Published IMU: {raw}")
                # print("OK")

    def parse_to_imu(self, data: str) -> Imu:
        """
        IMU 데이터 문자열을 sensor_msgs/Imu 메시지로 변환
        - JSON 예시: {"ax":0.1,"ay":0.2,"az":9.8,"gx":0.01,"gy":0.02,"gz":0.03}
        - CSV 예시 : ID,time,ax,ay,az,gx,gy,gz,roll,pitch,yaw
        """
        imu = Imu()

        try:
            d = json.loads(data)
        except Exception:
            parts = [p.strip() for p in data.split(',')]
            keys = ['ID', 'time', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'roll', 'pitch', 'yaw']
            d = dict(zip(keys, map(float, parts)))

        # Euler -> Quaternion
        roll=float(d.get('roll',0.0))
        pitch=float(d.get('pitch',0.0))
        yaw=float(d.get('yaw',0.0))
        r=R.from_euler('xyz',[roll,pitch,yaw],degrees=True)
        q=r.as_quat()
        imu.orientation.x = q[0]
        imu.orientation.y = q[1]
        imu.orientation.z = q[2]
        imu.orientation.w = q[3]
        
        imu.linear_acceleration.x = float(d.get('ax', 0.0))
        imu.linear_acceleration.y = float(d.get('ay', 0.0))
        imu.linear_acceleration.z = float(d.get('az', 0.0))
        imu.angular_velocity.x = float(d.get('gx', 0.0))
        imu.angular_velocity.y = float(d.get('gy', 0.0))
        imu.angular_velocity.z = float(d.get('gz', 0.0))
        imu.orientation_covariance[0] = -1.0  # orientation 미사용

        return imu


def main():
    rclpy.init()
    node = RedisPubSubToIMU(
        host='127.0.0.1',
        port=6379,
        channel='imu_data',
        frame_id='imu_link'
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
