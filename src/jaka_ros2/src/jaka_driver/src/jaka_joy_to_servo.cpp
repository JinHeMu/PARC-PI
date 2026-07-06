/*********************************************************************
 * joystick_servo_example.cpp  (DH AG95 夹爪 + 扳机增量 + 加速度限幅)
 *
 *  控制映射:
 *    - 左摇杆 X / Y      : 末端姿态 angular.x / angular.y
 *    - LB / RB           : 末端姿态 angular.z (yaw)
 *    - 右摇杆 Y          : 末端上下 linear.z
 *    - D-Pad Y (上下)    : 末端水平 linear.x (前 / 后)
 *    - D-Pad X (左右)    : 末端水平 linear.y (左 / 右)
 *    - LT 扳机           : 夹爪逐步闭合 (位置递减)
 *    - RT 扳机           : 夹爪逐步张开 (位置递增)
 *    - CHANGE_VIEW / MENU: 切换参考坐标系
 *
 *  Twist 速度采用 "加速度限幅" 替代原本的低通滤波:
 *    每个分量的变化量受 |Δv| <= a_max * dt 限制,
 *    启动 / 停止都是线性的, 没有指数尾巴, 抖动更小.
 *********************************************************************/

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>


#include <rclcpp/client.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/subscription.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/utilities.hpp>

// ==========================
// 话题名
// ==========================

const std::string JOY_TOPIC     = "/joy";
const std::string TWIST_TOPIC   = "/servo_node/delta_twist_cmds";
const std::string GRIPPER_TOPIC = "/gripper/ctrl";

const std::string EEF_FRAME_ID  = "gripper_center_link";
const std::string BASE_FRAME_ID = "base_link";

// ==========================
// 夹爪参数 (DH AG95: 0=全闭, 100=全开)
// ==========================

const double GRIPPER_MIN_POS  = 0.0;     // 全闭
const double GRIPPER_MAX_POS  = 100.0;   // 全开
const double GRIPPER_INIT_POS = 100.0;   // 启动初值 (默认张开)
const double GRIPPER_FORCE    = 10.0;    // 夹持力 (%)
const double GRIPPER_SPEED    = 50.0;    // 速度 (%)

// 扳机完全压下时的目标位置变化率, 单位: % / 秒
const double GRIPPER_RATE = 100.0;

// 扳机死区, 防止零位附近抖动
const double TRIGGER_DEADBAND = 0.03;

// 累计位置变化超过该阈值才下发新命令, 避免对夹爪驱动刷屏
const double GRIPPER_PUBLISH_DELTA = 0.5;

// ==========================
// 速度限幅参数 (代替低通滤波)
// ==========================

// 摇杆 / D-Pad 死区
const double JOY_DEADBAND = 0.05;

// 加速度上限: 单位归一化速度 / 秒
//
//   0 -> 1 所需时间 ≈ 1 / a_max
//   越大响应越快, 越小越平滑
//
// 推荐调参:
//   - 想再软一点: 把 LIN 降到 2~3, ANG 降到 3~4
//   - 想再硬一点: 把 LIN 加到 6~8, ANG 加到 8~12
const double MAX_LIN_ACCEL = 6.0;
const double MAX_ANG_ACCEL = 8.0;

// 双边都接近零时直接吸到 0, 避免浮点尾巴
const double ZERO_EPS = 1e-4;

// ==========================
// XBOX One 手柄映射
// ==========================

enum Axis
{
  LEFT_STICK_X = 0,
  LEFT_STICK_Y = 1,
  LEFT_TRIGGER = 2,
  RIGHT_STICK_X = 3,
  RIGHT_STICK_Y = 4,
  RIGHT_TRIGGER = 5,
  D_PAD_X = 6,
  D_PAD_Y = 7
};

enum Button
{
  A = 0,
  B = 1,
  X = 2,
  Y = 3,
  LEFT_BUMPER = 4,
  RIGHT_BUMPER = 5,
  CHANGE_VIEW = 6,
  MENU = 7,
  HOME = 8,
  LEFT_STICK_CLICK = 9,
  RIGHT_STICK_CLICK = 10
};

// 扳机轴未按下时的默认值 (大多数驱动是 1.0)
std::map<Axis, double> AXIS_DEFAULTS = {
  { LEFT_TRIGGER, 1.0 },
  { RIGHT_TRIGGER, 1.0 }
};

// ==========================
// 工具函数
// ==========================

double getAxisValue(const std::vector<float>& axes, Axis axis)
{
  size_t index = static_cast<size_t>(axis);

  if (index < axes.size())
    return static_cast<double>(axes[index]);

  if (AXIS_DEFAULTS.count(axis))
    return AXIS_DEFAULTS.at(axis);

  return 0.0;
}

int getButtonValue(const std::vector<int>& buttons, Button button)
{
  size_t index = static_cast<size_t>(button);

  if (index < buttons.size())
    return buttons[index];

  return 0;
}

double applyDeadband(double v)
{
  return std::abs(v) < JOY_DEADBAND ? 0.0 : v;
}

/**
 * 根据手柄状态生成 "目标 Twist".
 *   该函数只做映射 + 死区, 不做平滑;
 *   平滑由后面的加速度限幅完成.
 */
void buildTargetTwist(const std::vector<float>& axes,
                      const std::vector<int>& buttons,
                      geometry_msgs::msg::Twist& twist)
{
  // ---- 平移 ----
  //
  // D-Pad:
  //   D_PAD_Y (上推 = +1) -> linear.x (前)
  //   D_PAD_X 不同驱动方向可能相反, 如果发现 左右是反的, 把这里加个负号
  twist.linear.y = getAxisValue(axes, D_PAD_Y);
  twist.linear.x = -getAxisValue(axes, D_PAD_X);

  // 右摇杆 Y: 末端上下
  twist.linear.z = getAxisValue(axes, RIGHT_STICK_Y);

  // ---- 姿态 ----
  twist.angular.y = -getAxisValue(axes, LEFT_STICK_X);
  twist.angular.x = getAxisValue(axes, LEFT_STICK_Y);

  double yaw_pos = static_cast<double>(getButtonValue(buttons, RIGHT_BUMPER));
  double yaw_neg = -1.0 * static_cast<double>(getButtonValue(buttons, LEFT_BUMPER));
  twist.angular.z = yaw_pos + yaw_neg;

  // ---- 死区 ----
  twist.linear.x  = applyDeadband(twist.linear.x);
  twist.linear.y  = applyDeadband(twist.linear.y);
  twist.linear.z  = applyDeadband(twist.linear.z);
  twist.angular.x = applyDeadband(twist.angular.x);
  twist.angular.y = applyDeadband(twist.angular.y);
  twist.angular.z = applyDeadband(twist.angular.z);
}

void updateCmdFrame(std::string& frame_name, const std::vector<int>& buttons)
{
  if (getButtonValue(buttons, CHANGE_VIEW) && frame_name == EEF_FRAME_ID)
  {
    frame_name = BASE_FRAME_ID;
  }
  else if (getButtonValue(buttons, MENU) && frame_name == BASE_FRAME_ID)
  {
    frame_name = EEF_FRAME_ID;
  }
}

namespace moveit_servo
{

class JoyToServoPub : public rclcpp::Node
{
public:
  JoyToServoPub(const rclcpp::NodeOptions& options)
    : Node("joy_to_twist_publisher", options),
      frame_to_publish_(BASE_FRAME_ID),
      gripper_target_position_(GRIPPER_INIT_POS),
      last_published_gripper_position_(GRIPPER_INIT_POS)
  {
    joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        JOY_TOPIC,
        rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Joy::ConstSharedPtr& msg)
        {
          joyCB(msg);
        });

    twist_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
        TWIST_TOPIC, rclcpp::SystemDefaultsQoS());


    servo_start_client_ = this->create_client<std_srvs::srv::Trigger>(
        "/servo_node/start_servo");

    if (servo_start_client_->wait_for_service(std::chrono::seconds(1)))
    {
      servo_start_client_->async_send_request(
          std::make_shared<std_srvs::srv::Trigger::Request>());
    }
    else
    {
      RCLCPP_WARN(this->get_logger(),
                  "Service /servo_node/start_servo is not available.");
    }
  }

  ~JoyToServoPub() override = default;

  // ==========================
  // 夹爪控制
  // ==========================

  /**
   * 通过扳机做增量式夹爪控制:
   *   RT 越按越深 -> 夹爪逐步张开 (target +)
   *   LT 越按越深 -> 夹爪逐步闭合 (target -)
   *   松手 -> 速率归零, 当前位置保持
   */
  void updateGripperFromTriggers(const std::vector<float>& axes, double dt)
  {
    double rt = getAxisValue(axes, RIGHT_TRIGGER);
    double lt = getAxisValue(axes, LEFT_TRIGGER);

    // 扳机原始范围: 1.0 (松) -> -1.0 (压到底)
    // 归一化为 "压下量" [0, 1]
    double rt_press = (AXIS_DEFAULTS.at(RIGHT_TRIGGER) - rt) * 0.5;
    double lt_press = (AXIS_DEFAULTS.at(LEFT_TRIGGER)  - lt) * 0.5;

    rt_press = std::clamp(rt_press, 0.0, 1.0);
    lt_press = std::clamp(lt_press, 0.0, 1.0);

    if (rt_press < TRIGGER_DEADBAND) rt_press = 0.0;
    if (lt_press < TRIGGER_DEADBAND) lt_press = 0.0;

    double rate  = (rt_press - lt_press) * GRIPPER_RATE;  // % / s
    double delta = rate * dt;

    if (std::abs(delta) < 1e-6)
      return;

    double new_target = std::clamp(
        gripper_target_position_ + delta,
        GRIPPER_MIN_POS,
        GRIPPER_MAX_POS);

    // 刚刚撞到端点 -> 强制下发一次, 确保夹爪真正贴到极限
    bool just_hit_limit =
        (new_target <= GRIPPER_MIN_POS + 1e-3 &&
         last_published_gripper_position_ > GRIPPER_MIN_POS + 1e-3) ||
        (new_target >= GRIPPER_MAX_POS - 1e-3 &&
         last_published_gripper_position_ < GRIPPER_MAX_POS - 1e-3);

    gripper_target_position_ = new_target;

    if (just_hit_limit ||
        std::abs(gripper_target_position_ -
                 last_published_gripper_position_) >= GRIPPER_PUBLISH_DELTA)
    {
      last_published_gripper_position_ = gripper_target_position_;
    }
  }

  // ==========================
  // 加速度限幅 (取代低通滤波)
  // ==========================

  /**
   * 把 current 朝 target 推进, 单步变化不超过 max_accel * dt.
   *   - 起步: 输出线性加速
   *   - 松手: 输出线性减速, 有限时间内精确到 0
   *   - 不再有低通滤波那种永远趋近的指数尾巴
   */
  static double rateLimit(double target,
                          double current,
                          double max_accel,
                          double dt)
  {
    double max_step = max_accel * dt;
    double diff = target - current;

    if (diff >  max_step) diff =  max_step;
    if (diff < -max_step) diff = -max_step;

    double result = current + diff;

    // 目标与输出都极接近 0 -> 直接归零, 防止浮点残留
    if (std::abs(target) < ZERO_EPS && std::abs(result) < ZERO_EPS)
      result = 0.0;

    return result;
  }

  void applyAccelLimit(geometry_msgs::msg::Twist& msg, double dt)
  {
    twist_cmd_.linear.x =
        rateLimit(msg.linear.x, twist_cmd_.linear.x, MAX_LIN_ACCEL, dt);
    twist_cmd_.linear.y =
        rateLimit(msg.linear.y, twist_cmd_.linear.y, MAX_LIN_ACCEL, dt);
    twist_cmd_.linear.z =
        rateLimit(msg.linear.z, twist_cmd_.linear.z, MAX_LIN_ACCEL, dt);

    twist_cmd_.angular.x =
        rateLimit(msg.angular.x, twist_cmd_.angular.x, MAX_ANG_ACCEL, dt);
    twist_cmd_.angular.y =
        rateLimit(msg.angular.y, twist_cmd_.angular.y, MAX_ANG_ACCEL, dt);
    twist_cmd_.angular.z =
        rateLimit(msg.angular.z, twist_cmd_.angular.z, MAX_ANG_ACCEL, dt);

    msg = twist_cmd_;
  }

  // ==========================
  // Joy 回调
  // ==========================

  void joyCB(const sensor_msgs::msg::Joy::ConstSharedPtr& msg)
  {
    // 统一计算回调间隔, 给加速度限幅和夹爪增量共用
    rclcpp::Time now = this->now();
    double dt = 0.1;  // 默认按 50 Hz

    if (last_callback_time_.nanoseconds() > 0)
    {
      dt = (now - last_callback_time_).seconds();
      dt = std::clamp(dt, 0.001, 0.1);
    }
    last_callback_time_ = now;

    // 1) 扳机驱动的夹爪增量
    updateGripperFromTriggers(msg->axes, dt);

    // 2) Twist
    updateCmdFrame(frame_to_publish_, msg->buttons);

    auto twist_msg = std::make_unique<geometry_msgs::msg::TwistStamped>();

    // 先映射出目标 Twist, 再用加速度限幅推进 twist_cmd_
    buildTargetTwist(msg->axes, msg->buttons, twist_msg->twist);
    applyAccelLimit(twist_msg->twist, dt);

    twist_msg->header.frame_id = frame_to_publish_;
    twist_msg->header.stamp    = this->now();

    twist_pub_->publish(std::move(twist_msg));
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr servo_start_client_;

  std::string frame_to_publish_;

  // 夹爪状态
  double gripper_target_position_;
  double last_published_gripper_position_;

  // 回调时间戳
  rclcpp::Time last_callback_time_;

  // 已下发的 Twist (加速度限幅器的状态量)
  geometry_msgs::msg::Twist twist_cmd_;
};

}  // namespace moveit_servo

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(moveit_servo::JoyToServoPub)

