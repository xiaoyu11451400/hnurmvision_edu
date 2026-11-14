#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

// 第二讲的相机显示窗口：检测框和置信度直接画在这张画面的 Mat 上再输出
constexpr const char* kViewWindowName = "RoboMaster training";
constexpr int kKeyEsc = 27;
}

int main(int argc, char** argv) {
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  // 每次 cv::waitKey 都会清空键盘队列，所以只在主循环里统一读一次，再分发给各窗口
  bool quit = false;

  while (!quit && !interrupted) {
    auto input = devices.read();
    training::Command command{};
    if (input) {
      const auto armors = training::lesson03_detection(input->image);
      // 先跟踪：PnP 使用未绘制的原始图像，绘制不会影响解算
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);

      // 检测框/置信度直接画在第二讲订阅得到的原始相机图像上（原地绘制）
      training::lesson03_draw_armors(input->image, armors);
    }
    if (interrupted) break;
    devices.write(command);

    if (input && devices.config.display) {
      cv::imshow(kViewWindowName, input->image);
    }

    // ---------------- 键盘监听 ----------------
    const int key = cv::waitKey(1);
    if (key == kKeyEsc) {
      quit = true;
    } else if (key == 'q' || key == 'Q') {
      quit = true;
    }
  }

  // 干净退出：销毁所有 OpenCV 窗口（包括用户点 'X' 留下的僵尸窗口）
  cv::destroyAllWindows();
  devices.close();
  return 0;
}
