#pragma once
#include "config.hpp"
#include "types.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/context.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/subscription.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>


namespace training::io {
struct IoContext {
  explicit IoContext(Config cfg) : config( std::move(cfg) ){}

  Config config;
  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

  using RecvMsg = hnurm_interfaces::msg::VisionRecvData;
private:
  using ImageMsg = sensor_msgs::msg::Image;
  
  using SendMsg = hnurm_interfaces::msg::VisionSendData;

  void on_image( ImageMsg::ConstSharedPtr msg);
  void on_feedback( RecvMsg::ConstSharedPtr msg);
  void publish_stop();

  rclcpp::Context::SharedPtr context_;
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  rclcpp::Subscription<ImageMsg>::SharedPtr image_sub_;
  rclcpp::Subscription<RecvMsg>::SharedPtr recv_sub_;
  rclcpp::Publisher<SendMsg>::SharedPtr send_pub_;

  mutable std::mutex mutex_;
  TimePoint feedback_time_{};
  cv::Mat latest_image_; //CV_8UC3 BGR，像素内存与 ROS 消息无关
  TimePoint latest_image_time_{};

  std::uint64_t image_seq_ = 0;
  std::uint64_t delivered_seq_ = 0;

  bool feedback_valid_ = false;
  double gimbal_yaw_deg_ = 0.0;
  double gimbal_pitch_deg_ = 0.0;
  double gimbal_roll_deg_ = 0.0;
  double bullet_speed_mps_ = 0.0;
  std::uint8_t work_mode_ = 0;
  double mount_pitch_deg_ = 25.0; //由 camera_to_gimbal 推得，仿真车模为 25
};
}  // namespace training::io

//main.cpp 里是 IoContext devices{config};，显式构造函数接受 Config，花括号初始化仍然可用。