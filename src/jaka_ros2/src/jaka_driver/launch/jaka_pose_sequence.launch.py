from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="tracer_jaka_zu5",
            package_name="tracer_jaka_moveit_config",
        )
        .robot_description()
        .robot_description_semantic()
        .robot_description_kinematics()
        .planning_pipelines(pipelines=["ompl"])
        .trajectory_execution()
        .to_moveit_configs()
    )

    jaka_pose_sequence_node = Node(
        package="jaka_driver",
        executable="jaka_pose_sequence",
        name="jaka_pose_sequence",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "group_name": "jaka_arm",
                "execute_motion": True,
                "velocity_scaling": 0.02,
                "acceleration_scaling": 0.02,
                "planning_time": 10.0,
                "planning_attempts": 10,
                "pause_ms": 1000,
            },
        ],
    )

    return LaunchDescription([
        jaka_pose_sequence_node,
    ])
