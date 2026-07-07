#!/usr/bin/env python3
"""
ROS2 数据采集节点：记录四元数、关节角、关节速度、六维力/力矩到 CSV。

CSV 列顺序：
    time_sec,
    qx, qy, qz, qw,
    joint_1_pos ... joint_6_pos,
    joint_1_vel ... joint_6_vel,
    fx, fy, fz, tx, ty, tz
"""
import csv

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

from tf2_ros import Buffer, TransformListener


class PayloadDataCollector(Node):
    def __init__(self):
        super().__init__('payload_data_collector')

        self.csv_filename = 'payload_joint_wrench_quat_data.csv'

        # TF 坐标系：lookup_transform(sensor_frame, base_frame)
        # 与离线辨识代码保持一致
        self.base_frame = 'base_link'
        self.sensor_frame = 'jk_se_vi_200_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_names = [f'joint_{i}' for i in range(1, 7)]

        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_wrench = None

        self.record_count = 0
        self.tf_fail_count = 0

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.wrench_sub = self.create_subscription(
            WrenchStamped, '/jaka_fts_broadcaster/wrench', self.wrench_callback, 10)

        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        header = ['time_sec']
        header += ['qx', 'qy', 'qz', 'qw']
        header += [f'{name}_pos' for name in self.joint_names]
        header += [f'{name}_vel' for name in self.joint_names]
        header += ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        self.csv_writer.writerow(header)

        # 采集频率 10 Hz（timer 周期 0.1 s）
        self.timer = self.create_timer(0.1, self.record_data)

        self.get_logger().info(f"Data Collector Started. Saving to {self.csv_filename}")
        self.get_logger().info("Recording quaternion, joint positions/velocities, and F/T.")
        self.get_logger().info("Press Ctrl+C to stop recording.")

    def joint_state_callback(self, msg: JointState):
        """/joint_states 关节顺序不固定，按名字重排为 joint_1 ~ joint_6。"""
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        positions, velocities = [], []
        for joint_name in self.joint_names:
            if joint_name not in name_to_index:
                self.get_logger().warn(f"Joint {joint_name} not found in /joint_states")
                return
            idx = name_to_index[joint_name]
            if idx >= len(msg.position):
                self.get_logger().warn(f"Position for {joint_name} not available")
                return
            positions.append(msg.position[idx])
            velocities.append(msg.velocity[idx] if len(msg.velocity) > idx else 0.0)

        self.current_joint_positions = positions
        self.current_joint_velocities = velocities

    def wrench_callback(self, msg: WrenchStamped):
        self.current_wrench = msg.wrench

    def get_current_quaternion(self):
        """获取 base_frame -> sensor_frame 的旋转四元数。"""
        trans = self.tf_buffer.lookup_transform(
            self.sensor_frame, self.base_frame, Time())
        r = trans.transform.rotation
        return [r.x, r.y, r.z, r.w]

    def record_data(self):
        if (self.current_joint_positions is None
                or self.current_joint_velocities is None
                or self.current_wrench is None):
            return

        try:
            quat = self.get_current_quaternion()
        except Exception as e:
            self.tf_fail_count += 1
            if self.tf_fail_count % 50 == 1:
                self.get_logger().warn(
                    f"Could not get TF {self.base_frame} -> {self.sensor_frame}: {e}")
            return

        time_sec = self.get_clock().now().nanoseconds * 1e-9
        w = self.current_wrench

        row = [time_sec]
        row += quat
        row += self.current_joint_positions
        row += self.current_joint_velocities
        row += [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z]

        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.record_count += 1

        self.get_logger().info(
            f"Point {self.record_count} | "
            f"q=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f}) | "
            f"J1={self.current_joint_positions[0]:.3f} rad | "
            f"Fz={w.force.z:.2f} N")

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f"File closed. Total points: {self.record_count}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PayloadDataCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
