#include "io_context.hpp"

#include <hnurm_interfaces/msg/bullet_speed.hpp>
#include <hnurm_interfaces/msg/target_state.hpp>
#include <hnurm_interfaces/msg/work_mode.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/time.hpp>

#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

namespace training::io {
namespace {
using rclcpp::SensorDataQoS;

constexpr int kSpinTimeoutMs = 10;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kMaxInputAgeSec = 0.2;  // 输入过期阈值
constexpr double kGimbalLimitDeg = 45.0; // 仿真器对协议俯仰角的机械限位

// 把消息里的 ROS 时间戳换算成“距离现在多久”，再映射到程序 Clock。
// 仿真器用的是 SystemTime；若换成仿真时间，这里会返回 nullopt，
// 由调用方用收到消息的时刻兜底，至少保证帧间差分是单调的。
std::optional<TimePoint> map_ros_stamp(const builtin_interfaces::msg::Time& stamp) {
  const auto ros_as_system =
      std::chrono::system_clock::time_point{std::chrono::seconds(stamp.sec) +
                                            std::chrono::nanoseconds(stamp.nanosec)};
  const auto offset = std::chrono::system_clock::now() - ros_as_system;
  const auto magnitude =
      offset < std::chrono::system_clock::duration::zero() ? -offset : offset;
  if (magnitude > std::chrono::hours(24)) return std::nullopt;
  return Clock::now() -
         std::chrono::duration_cast<Clock::duration>(magnitude);
}
}  // namespace

void IoContext::open() {
  context_ = std::make_shared<rclcpp::Context>();
  context_->init(0, nullptr);

  rclcpp::NodeOptions options;
  options.context(context_);
  node_ = std::make_shared<rclcpp::Node>("training_io", options);

  rclcpp::ExecutorOptions exec_options;
  exec_options.context = context_;
  executor_ =
      std::make_unique<rclcpp::executors::SingleThreadedExecutor>(exec_options);

  // 传感器通道：尽力投递 + 只留最新。仿真器 /image_raw 用默认 QoS（reliable），
  // best_effort 订阅与 reliable 发布是兼容的。
  const auto qos = SensorDataQoS().keep_last(1);

  image_sub_ = node_->create_subscription<ImageMsg>(
      "/image_raw", qos,
      [this](ImageMsg::ConstSharedPtr msg) { on_image(std::move(msg)); });
  recv_sub_ = node_->create_subscription<RecvMsg>(
      "/vision_receive_data", qos,
      [this](RecvMsg::ConstSharedPtr msg) { on_feedback(std::move(msg)); });
  send_pub_ =
      node_->create_publisher<SendMsg>("/vision_send_data", qos);

  executor_->add_node(node_);
}

void IoContext::on_image(ImageMsg::ConstSharedPtr msg) {
  if (!msg || msg->width == 0 || msg->height == 0) return;
  const int rows = static_cast<int>(msg->height);
  const int cols = static_cast<int>(msg->width);

  if (msg->encoding != "bgr8" && msg->encoding != "rgb8") {
    return;  // 仿真器固定 rgb8；算法需要 3 通道 8 位图像
  }

  // 用消息内存做只读视图，随后立即复制/转换，算法拿到的内存与 ROS 消息无关。
  const cv::Mat view(rows, cols, CV_8UC3,
                     const_cast<unsigned char*>(msg->data.data()),
                     static_cast<std::size_t>(msg->step));
  cv::Mat image;
  if (msg->encoding == "rgb8") {
    cv::cvtColor(view, image, cv::COLOR_RGB2BGR);
  } else {
    view.copyTo(image);
  }

  auto stamp = map_ros_stamp(msg->header.stamp);
  std::lock_guard<std::mutex> lock(mutex_);
  latest_image_time_ = stamp.value_or(Clock::now());
  latest_image_ = std::move(image);
  ++image_seq_;
}

void IoContext::on_feedback(RecvMsg::ConstSharedPtr msg) {
  if (!msg) return;
  std::lock_guard<std::mutex> lock(mutex_);
  feedback_valid_ = true;
  feedback_time_ =
      map_ros_stamp(msg->header.stamp).value_or(Clock::now());
  gimbal_yaw_deg_ = static_cast<double>(msg->yaw);
  gimbal_pitch_deg_ = static_cast<double>(msg->pitch);
  gimbal_roll_deg_ = static_cast<double>(msg->roll);
  bullet_speed_mps_ = static_cast<double>(msg->bullet_speed.data);
  work_mode_ = msg->work_mode.data;
}

std::optional<SensorData> IoContext::read() {
  if (!executor_) return std::nullopt;
  // 给回调一次执行机会；有超时，Ctrl+C 不会被卡住。
  executor_->spin_some(std::chrono::milliseconds(kSpinTimeoutMs));

  std::lock_guard<std::mutex> lock(mutex_);
  if (image_seq_ == delivered_seq_ || latest_image_.empty())
    return std::nullopt;

  const double feedback_age =
      std::chrono::duration<double>(Clock::now() - feedback_time_).count();
  if (!feedback_valid_ || feedback_age > kMaxInputAgeSec ||
      bullet_speed_mps_ <= 0.0) {
    // 没有姿态或弹速就构造不出完整 SensorData；丢弃该帧。
    delivered_seq_ = image_seq_;
    return std::nullopt;
  }

  SensorData data;
  data.image = std::move(latest_image_);  // 转移所有权，像素内存独立于 IO 缓冲
  data.timestamp = latest_image_time_;
  data.bullet_speed = bullet_speed_mps_;

  // 云台坐标系：x 前、y 左、z 上；yaw 绕 z，pitch 绕 y，roll 绕 x。
  // 符号说明见第 5 节。
  const double yaw = gimbal_yaw_deg_ * kDegToRad;
  const double pitch = gimbal_pitch_deg_ * kDegToRad;
  const double roll = gimbal_roll_deg_ * kDegToRad;
  data.imu_q =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(-roll, Eigen::Vector3d::UnitX());
  data.imu_q.normalize();

  delivered_seq_ = image_seq_;
  return data;
}

void IoContext::publish_stop() {
  if (!send_pub_) return;
  SendMsg msg;
  msg.header.stamp = node_ ? node_->now() : rclcpp::Time();
  msg.header.frame_id = "gimbal_link";
  msg.target_state.data =
      hnurm_interfaces::msg::TargetState::LOST_TARGET;
  msg.pitch = 0.0f;
  msg.yaw = 0.0f;
  msg.target_distance = 0.0f;
  send_pub_->publish(msg);
}

void IoContext::write(const Command& command) {
  if (!send_pub_) return;

  bool auto_aim;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const double feedback_age =
        std::chrono::duration<double>(Clock::now() - feedback_time_).count();
    auto_aim = feedback_valid_ && feedback_age <= kMaxInputAgeSec &&
               work_mode_ ==
                   hnurm_interfaces::msg::WorkMode::AUTO_AIM;
  }

  const bool angles_ok =
      std::isfinite(command.yaw) && std::isfinite(command.pitch);
  if (!command.control || !auto_aim || !angles_ok) {
    publish_stop();
    return;
  }

  const bool fire = command.shoot && config.allow_fire;
  SendMsg msg;
  msg.header.stamp = node_->now();
  msg.header.frame_id = "gimbal_link";
  msg.target_state.data =
      fire ? hnurm_interfaces::msg::TargetState::FIRE
           : hnurm_interfaces::msg::TargetState::CONVERGING;
  msg.yaw = static_cast<float>(command.yaw * kRadToDeg);

  // Command.pitch 是 IMU 系中枪管的绝对仰角；协议里的 pitch 是云台本体俯仰角。
  // 车模的相机/枪管相对云台有 25° 固定装配仰角：protocol_pitch = mount - elevation。
  const double protocol_pitch_deg =
      mount_pitch_deg_ - command.pitch * kRadToDeg;
  if (std::abs(protocol_pitch_deg) > kGimbalLimitDeg) {
    publish_stop();  // 超出机械限位就放手
    return;
  }
  msg.pitch = static_cast<float>(protocol_pitch_deg);

  // 仿真器把 target_distance == -1 当成停止；控制时必须给非 -1 的值。
  msg.target_distance = 0.0f;
  send_pub_->publish(msg);
}

void IoContext::close() {
  publish_stop();
  // 给 DDS 一点时间把停止指令送出去，再结束上下文。
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  if (context_ && context_->is_valid()) {
    context_->shutdown("io closed");
  }
  if (executor_) executor_.reset();
  image_sub_.reset();
  recv_sub_.reset();
  send_pub_.reset();
  node_.reset();
  context_.reset();

  std::lock_guard<std::mutex> lock(mutex_);
  latest_image_.release();
  feedback_valid_ = false;
}
}  // namespace training::io
