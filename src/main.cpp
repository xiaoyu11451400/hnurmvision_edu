#include "lessons.hpp"
#include "io_context.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }

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
    const auto input = devices.read();
    training::Command command{};
    if (input) {
      const auto armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
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
    } else if (key == 'v' || key == 'V') {
      const bool enable = !training::lesson03_draw_enabled();
      if (enable) {
        // 先切回开启状态，下一帧 lesson03 会重新 imshow 并创建同名窗口
        training::lesson03_set_draw(true);
        cv::namedWindow(training::kDetectionWindowName, cv::WINDOW_AUTOSIZE);
      } else {
        // 先销毁窗口再通知 lesson03 停止显示，避免本帧之后残留空窗口
        cv::destroyWindow(training::kDetectionWindowName);
        training::lesson03_set_draw(false);
      }
    }
  }

  // 干净退出：销毁所有 OpenCV 窗口（包括用户点 'X' 留下的僵尸窗口）
  cv::destroyAllWindows();
  devices.close();
  return 0;
}
