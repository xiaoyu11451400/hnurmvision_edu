#pragma once
#include "types.hpp"
#include "tracking_state.hpp"
#include <optional>
#include <vector>

namespace training {
// 第三讲可视化窗口名（main 退出时用它销毁窗口）
inline constexpr const char* kDetectionWindowName = "Detection";

std::vector<Armor> lesson03_detection(const cv::Mat& image);
// 运行时开关：是否绘制并显示检测结果
void lesson03_set_draw(bool enabled);
bool lesson03_draw_enabled();
std::optional<Target> lesson04_tracking(const std::vector<Armor>& armors,
    TimePoint timestamp, const Eigen::Quaterniond& imu_q, const CameraCalibration& calibration, TrackingState& state);
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed);
}  // namespace training
