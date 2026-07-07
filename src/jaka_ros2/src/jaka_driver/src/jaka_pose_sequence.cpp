#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/move_it_error_codes.hpp>

template <typename T>
T get_parameter_or_declare(
  const rclcpp::Node::SharedPtr& node,
  const std::string& name,
  const T& default_value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, default_value);
  }

  T value;
  node->get_parameter(name, value);
  return value;
}

struct JointPose
{
  std::string name;
  std::map<std::string, double> joints;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>(
    "jaka_pose_sequence",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  auto logger = rclcpp::get_logger("jaka_pose_sequence");

  const std::string planning_group =
    get_parameter_or_declare<std::string>(node, "group_name", "jaka_arm");

  const bool execute_motion =
    get_parameter_or_declare<bool>(node, "execute_motion", true);

  const double velocity_scaling =
    get_parameter_or_declare<double>(node, "velocity_scaling", 0.15);

  const double acceleration_scaling =
    get_parameter_or_declare<double>(node, "acceleration_scaling", 0.15);

  const double planning_time =
    get_parameter_or_declare<double>(node, "planning_time", 10.0);

  const int planning_attempts =
    get_parameter_or_declare<int>(node, "planning_attempts", 10);

  const int pause_ms =
    get_parameter_or_declare<int>(node, "pause_ms", 1000);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() {
    executor.spin();
  });

  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);

  move_group.setMaxVelocityScalingFactor(velocity_scaling);
  move_group.setMaxAccelerationScalingFactor(acceleration_scaling);
  move_group.setPlanningTime(planning_time);
  move_group.setNumPlanningAttempts(planning_attempts);
  move_group.allowReplanning(true);

  RCLCPP_INFO(logger, "Planning group: %s", planning_group.c_str());
  RCLCPP_INFO(logger, "Planning frame: %s", move_group.getPlanningFrame().c_str());
  RCLCPP_INFO(logger, "End effector link: %s", move_group.getEndEffectorLink().c_str());

  auto current_state = move_group.getCurrentState(10.0);
  if (!current_state) {
    RCLCPP_ERROR(logger, "Failed to receive current robot state. Check /joint_states.");
    executor.cancel();
    if (spinner.joinable()) {
      spinner.join();
    }
    rclcpp::shutdown();
    return 1;
  }

  const std::vector<JointPose> poses = {
    {
      "pose_00",
      {
        {"joint_1", -0.5236},
        {"joint_2",  1.5707},
        {"joint_3", -1.0647},
        {"joint_4",  1.1345},
        {"joint_5",  1.5707},
        {"joint_6",  0.2618}
      }
    },
    {
      "pose_01",
      {
        {"joint_1", 0.0},
        {"joint_2", 1.5707},
        {"joint_3", -1.5707},
        {"joint_4", 1.5707},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_02",
      {
        {"joint_1", 1.5707},
        {"joint_2", 1.0472},
        {"joint_3", 1.1345},
        {"joint_4", 1.0472},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_03",
      {
        {"joint_1", 1.5707},
        {"joint_2", 0.785398},
        {"joint_3", 1.6581},
        {"joint_4", 0.785398},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_04",
      {
        {"joint_1", 1.5707},
        {"joint_2", 1.9199},
        {"joint_3", -0.5236},
        {"joint_4", 0.785398},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_05",
      {
        {"joint_1", 1.5707},
        {"joint_2", 2.3562},
        {"joint_3", -1.5707},
        {"joint_4", 1.5707},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_06",
      {
        {"joint_1", 1.5707},
        {"joint_2", 1.309},
        {"joint_3", -1.309},
        {"joint_4", 1.5707},
        {"joint_5", 1.5707},
        {"joint_6", 0.785398}
      }
    },
    {
      "pose_07",  // 45°, 103°, -91°, 78°, 90°, 93°
      {
        {"joint_1",  0.785398},
        {"joint_2",  1.797689},
        {"joint_3", -1.588250},
        {"joint_4",  1.361357},
        {"joint_5",  1.570796},
        {"joint_6",  1.623156}
      }
    },
    {
      "pose_08",  // 144°, 100°, -104°, 94°, 90°, 21°
      {
        {"joint_1",  2.513274},
        {"joint_2",  1.745329},
        {"joint_3", -1.815142},
        {"joint_4",  1.640609},
        {"joint_5",  1.570796},
        {"joint_6",  0.366519}
      }
    },
    {
      "pose_09",  // 114°, 81°, -85°, 94°, 90°, -9°
      {
        {"joint_1",  1.989675},
        {"joint_2",  1.413717},
        {"joint_3", -1.483530},
        {"joint_4",  1.640609},
        {"joint_5",  1.570796},
        {"joint_6", -0.157080}
      }
    },
    {
      "pose_10",  // 76°, 100°, -145°, 180°, 100°, 104°
      {
        {"joint_1",  1.326450},
        {"joint_2",  1.745329},
        {"joint_3", -2.530727},
        {"joint_4",  3.141593},
        {"joint_5",  1.745329},
        {"joint_6",  1.815142}
      }
    },
    {
      "pose_11",  // 27°, 137°, -129°, 107°, 129°, 60°
      {
        {"joint_1",  0.471239},
        {"joint_2",  2.391101},
        {"joint_3", -2.251475},
        {"joint_4",  1.867502},
        {"joint_5",  2.251475},
        {"joint_6",  1.047198}
      }
    },
    {
      "pose_12",  // 90°, 130°, -111°, 116°, 90°, 0°
      {
        {"joint_1",  1.570796},
        {"joint_2",  2.268928},
        {"joint_3", -1.937315},
        {"joint_4",  2.024582},
        {"joint_5",  1.570796},
        {"joint_6",  0.000000}
      }
    },
    {
      "up",
      {
        {"joint_1", 0.0},
        {"joint_2", 1.5707},
        {"joint_3", 0.0},
        {"joint_4", 1.5707},
        {"joint_5", 3.14159},
        {"joint_6", 0.785398}
      }
    }
  };

  for (const auto& pose : poses) {
    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Planning to %s", pose.name.c_str());

    move_group.setStartStateToCurrentState();

    bool target_ok = move_group.setJointValueTarget(pose.joints);
    if (!target_ok) {
      RCLCPP_ERROR(logger, "Failed to set joint target for %s. Check joint names and group name.",
                   pose.name.c_str());
      break;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    auto plan_result = move_group.plan(plan);
    bool plan_success = (plan_result == moveit::core::MoveItErrorCode::SUCCESS);

    if (!plan_success) {
      RCLCPP_ERROR(logger, "Planning failed for %s", pose.name.c_str());
      break;
    }

    RCLCPP_INFO(logger, "Planning succeeded for %s", pose.name.c_str());

    if (execute_motion) {
      RCLCPP_INFO(logger, "Executing %s", pose.name.c_str());

      auto execute_result = move_group.execute(plan);
      bool execute_success = (execute_result == moveit::core::MoveItErrorCode::SUCCESS);

      if (!execute_success) {
        RCLCPP_ERROR(logger, "Execution failed for %s", pose.name.c_str());
        break;
      }

      RCLCPP_INFO(logger, "Execution finished for %s", pose.name.c_str());
      std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    } else {
      RCLCPP_WARN(logger, "execute_motion=false, only planned %s, not executed.",
                  pose.name.c_str());
    }
  }

  RCLCPP_INFO(logger, "Pose sequence finished.");

  executor.cancel();
  if (spinner.joinable()) {
    spinner.join();
  }

  rclcpp::shutdown();
  return 0;
}
