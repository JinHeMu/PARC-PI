#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped

from tf2_ros import Buffer, TransformListener

import csv


class PayloadDataCollector(Node):
    def __init__(self):
        super().__init__('payload_data_collector')

        self.csv_filename = 'payload_joint_wrench_quat_data.csv'

        # TF 坐标系
        # 保持和你原来离线辨识代码一致：
        # lookup_transform(sensor_frame, base_frame)
        self.base_frame = 'base_link'
        self.sensor_frame = 'jk_se_vi_200_link'

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 按 controller 配置中的关节名固定 CSV 输出顺序
        self.joint_names = [
            'joint_1',
            'joint_2',
            'joint_3',
            'joint_4',
            'joint_5',
            'joint_6'
        ]

        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_wrench = None

        self.record_count = 0
        self.tf_fail_count = 0

        # 订阅关节状态
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # 订阅六维力传感器
        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            '/jaka_fts_broadcaster/wrench',
            self.wrench_callback,
            10
        )

        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        header = ['time_sec']

        # 四元数：base_frame 到 sensor_frame 的旋转
        header += ['qx', 'qy', 'qz', 'qw']

        # 6 个关节位置
        header += [f'{name}_pos' for name in self.joint_names]

        # 6 个关节速度
        header += [f'{name}_vel' for name in self.joint_names]

        # 力和力矩
        header += ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']

        self.csv_writer.writerow(header)

        # 采集频率 50 Hz
        self.timer = self.create_timer(0.02, self.record_data)

        self.get_logger().info(f"Data Collector Started. Saving to {self.csv_filename}")
        self.get_logger().info("Recording quaternion, joint positions, joint velocities, and force/torque data.")
        self.get_logger().info("Press Ctrl+C to stop recording.")

    def joint_state_callback(self, msg: JointState):
        """
        /joint_states 中的关节顺序不一定固定，
        所以这里根据 joint name 重新排列成 joint_1 ~ joint_6。
        """
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        positions = []
        velocities = []

        for joint_name in self.joint_names:
            if joint_name not in name_to_index:
                self.get_logger().warn(f"Joint {joint_name} not found in /joint_states")
                return

            idx = name_to_index[joint_name]

            if idx >= len(msg.position):
                self.get_logger().warn(f"Position for {joint_name} not available")
                return

            positions.append(msg.position[idx])

            if len(msg.velocity) > idx:
                velocities.append(msg.velocity[idx])
            else:
                velocities.append(0.0)

        self.current_joint_positions = positions
        self.current_joint_velocities = velocities

    def wrench_callback(self, msg: WrenchStamped):
        self.current_wrench = msg.wrench

    def get_current_quaternion(self):
        """
        获取 base_link 到 jk_se_vi_200_link 的旋转四元数。

        注意：
        这里的顺序和你原始代码保持一致：
            lookup_transform(self.sensor_frame, self.base_frame, Time())
        """
        trans = self.tf_buffer.lookup_transform(
            self.sensor_frame,
            self.base_frame,
            Time()
        )

        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        return [qx, qy, qz, qw]

    def record_data(self):
        if self.current_joint_positions is None:
            return

        if self.current_joint_velocities is None:
            return

        if self.current_wrench is None:
            return

        try:
            quat = self.get_current_quaternion()
        except Exception as e:
            self.tf_fail_count += 1

            # 避免 50Hz 下刷屏，每 50 次失败提示一次
            if self.tf_fail_count % 50 == 1:
                self.get_logger().warn(
                    f"Could not get TF from {self.base_frame} to {self.sensor_frame}: {e}"
                )
            return

        now = self.get_clock().now()
        time_sec = now.nanoseconds * 1e-9

        fx = self.current_wrench.force.x
        fy = self.current_wrench.force.y
        fz = self.current_wrench.force.z

        tx = self.current_wrench.torque.x
        ty = self.current_wrench.torque.y
        tz = self.current_wrench.torque.z

        row = [time_sec]
        row += quat
        row += self.current_joint_positions
        row += self.current_joint_velocities
        row += [fx, fy, fz, tx, ty, tz]

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.record_count += 1

        self.get_logger().info(
            f"Recorded point {self.record_count} | "
            f"qx: {quat[0]:.4f}, qy: {quat[1]:.4f}, qz: {quat[2]:.4f}, qw: {quat[3]:.4f} | "
            f"J1: {self.current_joint_positions[0]:.4f} rad, "
            f"J1_vel: {self.current_joint_velocities[0]:.4f} rad/s, "
            f"Fz: {fz:.2f} N"
        )

    def destroy_node(self):
        self.csv_file.close()
        self.get_logger().info(f"File closed. Total points recorded: {self.record_count}")
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
