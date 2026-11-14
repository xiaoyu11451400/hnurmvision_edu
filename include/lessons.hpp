#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
std::vector<Armor> lesson03_detection(const cv::Mat& image);
// 把检测框和置信度直接绘制在传入的相机图像上（原地修改，不拷贝）
void lesson03_draw_armors(cv::Mat& image, const std::vector<Armor>& armors);
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
}  // namespace training
