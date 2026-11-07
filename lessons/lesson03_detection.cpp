#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "lessons.hpp"

namespace training {
namespace {

// ---------------- 模型与后处理配置 ----------------
const std::string MODEL_PATH = "model/best.xml";
constexpr float CONF_THRESHOLD = 0.5f;
constexpr float NMS_THRESHOLD = 0.4f;
constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;
constexpr int NUM_KEYPOINTS = 4;  // 每个装甲板 4 个角点，每个点 x,y 两个值

// 是否做 BGR -> RGB 通道交换。
// Ultralytics 训练时把 BGR 图像转成 RGB 送入网络，而 OpenVINO 只做计算、
// 不会自动交换通道，所以这里必须交换。
// 在验证集 200 张图上实测：交换后召回 85.8%，不交换只有 72.6%。
constexpr bool SWAP_RB = true;

// 录制/演示用的可视化开关；运行时可用 lesson03_set_draw 切换，不影响检测结果。
bool DRAW_RESULT = true;
bool WINDOW_CREATED = false;  // 是否已创建过窗口，用于判断用户是否点了关闭按钮
constexpr int VIEW_LINE_THICKNESS = 2;
constexpr double VIEW_FONT_SCALE = 0.6;
constexpr int VIEW_FONT_THICKNESS = 2;

// ---------------- 模型加载（整个进程只执行一次） ----------------
// 把 Core / 模型 / InferRequest 封装在一起。
// 注意成员声明顺序：core 必须比 compiled 先构造、后析构。
struct Detector {
  ov::Core core;
  ov::CompiledModel compiled;
  ov::InferRequest request;
  int box_length = 0;       // 每个预测框的输出长度
  int num_classes = 0;      // 类别数（从输出维度推导）
  int kpt_offset = 0;       // 关键点在输出中的起始行

  Detector() {
    // read_model 会自动识别 .xml + 同目录 .bin
    const std::shared_ptr<ov::Model> model = core.read_model(MODEL_PATH);
    compiled = core.compile_model(model, "CPU");
    request = compiled.create_infer_request();

    // 输出形状为 [1, box_length, num_boxes]，例如 [1, 26, 8400]
    const auto shape = compiled.output().get_partial_shape();
    if (shape.size() < 3 || !shape[1].is_static() || !shape[2].is_static()) {
      throw std::runtime_error("[lesson03] 模型输出形状必须是静态的 [1, N, 8400]");
    }
    box_length = static_cast<int>(shape[1].get_length());
    // 输出布局：4(xywh) + num_classes(各类别分数) + 8(4 个关键点 xy)
    num_classes = box_length - 4 - 2 * NUM_KEYPOINTS;
    kpt_offset = 4 + num_classes;
    if (num_classes <= 0) {
      throw std::runtime_error("[lesson03] 模型输出维度异常: " + std::to_string(box_length));
    }
    std::cerr << "[lesson03] 模型输出每行 " << box_length << " 个值（" << num_classes
              << " 类），关键点从索引 " << kpt_offset << " 开始\n";
  }
};

Detector& detector() {
  // C++11 起，函数内 static 局部变量的初始化是线程安全的
  static Detector instance;
  return instance;
}

// ---------------- 预处理：Letterbox ----------------
// 保持长宽比缩放，再把图像居中放进 640x640 的灰色画布。
// ratio 返回缩放比例，pad 返回左上角填充宽度，用于后处理逆映射。
void letterbox(const cv::Mat& image, cv::Mat& out, float& ratio, cv::Point2f& pad) {
  ratio = std::min(static_cast<float>(INPUT_W) / image.cols,
                   static_cast<float>(INPUT_H) / image.rows);
  const int new_w = static_cast<int>(std::round(image.cols * ratio));
  const int new_h = static_cast<int>(std::round(image.rows * ratio));

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  pad = cv::Point2f((INPUT_W - new_w) / 2.0f, (INPUT_H - new_h) / 2.0f);
  out = cv::Mat(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(static_cast<int>(pad.x), static_cast<int>(pad.y), new_w, new_h)));
}

// ---------------- IoU ----------------
float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
  const float inter_w = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
  const float inter_h = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);
  if (inter_w <= 0.f || inter_h <= 0.f) return 0.f;
  const float inter = inter_w * inter_h;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.f ? inter / uni : 0.f;
}

// ---------------- 可视化：四角连线 + 类别与置信度 ----------------
// 在 image 上原地绘制；corners 是浮点坐标，画线前转成整数像素。
void drawArmors(cv::Mat& image, const std::vector<Armor>& armors) {
  for (const Armor& armor : armors) {
    std::array<cv::Point, 4> pts{};
    for (int i = 0; i < 4; ++i) {
      pts[i] = cv::Point(cvRound(armor.corners[i].x), cvRound(armor.corners[i].y));
    }
    const cv::Scalar green(0, 255, 0);
    for (int i = 0; i < 4; ++i) {
      cv::line(image, pts[i], pts[(i + 1) % 4], green, VIEW_LINE_THICKNESS);
    }
    // 文字放在左上角点的左上方，保留两位小数
    char label[64];
    std::snprintf(label, sizeof(label), "%d %.2f", armor.class_id, armor.confidence);
    const cv::Point text_org(pts[0].x, std::max(pts[0].y - 8, 20));
    cv::putText(image, label, text_org, cv::FONT_HERSHEY_SIMPLEX, VIEW_FONT_SCALE, green,
                VIEW_FONT_THICKNESS);
  }
}

}  // namespace

// 运行时开关检测结果的可视化窗口；关闭时窗口会在 main 里销毁。
void lesson03_set_draw(bool enabled) {
  DRAW_RESULT = enabled;
  // 重新打开时当作从未创建过，防止误把旧窗口状态当成"用户已关闭"
  if (!enabled) WINDOW_CREATED = false;
}

bool lesson03_draw_enabled() { return DRAW_RESULT; }

// 第三讲：装甲板检测
// 输入：不可修改的 CV_8UC3 BGR 图像。
// 输出：Armor 列表；角点顺序为左上、右上、右下、左下，单位为像素。
// 图像坐标原点在左上；无法识别编号时 class_id=-1；没有装甲板时返回空列表。
std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  std::vector<Armor> armors;
  if (image.empty()) return armors;

  Detector& det = detector();

  // ---------------- 1. 图像预处理 ----------------
  cv::Mat letterboxed;
  float ratio = 1.f;
  cv::Point2f pad;
  letterbox(image, letterboxed, ratio, pad);

  cv::Mat rgb;
  if (SWAP_RB) {
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  } else {
    rgb = letterboxed;
  }

  // 归一化到 [0, 1] 并转为 float32
  cv::Mat f;
  rgb.convertTo(f, CV_32F, 1.0 / 255.0);

  // HWC -> NCHW：split 出 3 个通道，再依次拼进连续内存
  std::vector<cv::Mat> planes;
  cv::split(f, planes);
  std::vector<float> input_data(static_cast<size_t>(3) * INPUT_H * INPUT_W);
  const size_t plane_size = static_cast<size_t>(INPUT_H) * INPUT_W;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(input_data.data() + c * plane_size, planes[c].ptr<float>(),
                plane_size * sizeof(float));
  }

  // 用连续内存包装成 OpenVINO 张量，避免额外拷贝
  ov::Tensor input_tensor(ov::element::f32, {1, 3, INPUT_H, INPUT_W}, input_data.data());
  det.request.set_input_tensor(input_tensor);

  // ---------------- 2. 推理 ----------------
  det.request.infer();

  const ov::Tensor output = det.request.get_output_tensor();  // [1, box_length, num_boxes]
  const ov::Shape& shape = output.get_shape();
  const int num_boxes = static_cast<int>(shape[2]);
  const float* data = output.data<const float>();

  // 输出按行存放：第 row 个属性、第 i 个预测框位于 data[row * num_boxes + i]
  const auto at = [data, num_boxes](int row, int i) {
    return data[static_cast<size_t>(row) * num_boxes + i];
  };

  // ---------------- 3. 遍历预测框，按置信度过滤 ----------------
  struct Detection {
    cv::Rect2f box;                     // 640x640 letterbox 坐标系下的 xywh
    float confidence = 0.f;
    int class_id = -1;
    std::array<cv::Point2f, 4> kpts{};  // 640x640 letterbox 坐标系下的关键点
  };
  std::vector<Detection> detections;

  for (int i = 0; i < num_boxes; ++i) {
    // 索引 4 起是各类别分数（单类别模型时索引 4 就是置信度）。
    // 取最大类别分数作为该框的置信度和类别。
    float confidence = 0.f;
    int class_id = -1;
    for (int c = 0; c < det.num_classes; ++c) {
      const float score = at(4 + c, i);
      if (score > confidence) {
        confidence = score;
        class_id = c;
      }
    }
    if (confidence < CONF_THRESHOLD) continue;

    Detection d;
    d.confidence = confidence;
    d.class_id = class_id;

    // xywh（中心点 + 宽高）转 xywh（左上角 + 宽高）
    const float cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
    d.box = cv::Rect2f(cx - w / 2.f, cy - h / 2.f, w, h);

    // 索引 kpt_offset ~ kpt_offset+7 是 4 个关键点的 x,y
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
      d.kpts[k] = cv::Point2f(at(det.kpt_offset + 2 * k, i), at(det.kpt_offset + 2 * k + 1, i));
    }
    detections.push_back(d);
  }

  // ---------------- 4. NMS（非极大值抑制） ----------------
  // 按置信度从高到低排序，逐个保留，并抑制与其 IoU 超过阈值的低分框。
  std::vector<int> order(detections.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return detections[a].confidence > detections[b].confidence;
  });

  std::vector<bool> suppressed(detections.size(), false);
  std::vector<int> keep;
  keep.reserve(detections.size());
  for (size_t m = 0; m < order.size(); ++m) {
    const int idx = order[m];
    if (suppressed[idx]) continue;
    keep.push_back(idx);
    for (size_t n = m + 1; n < order.size(); ++n) {
      const int other = order[n];
      if (!suppressed[other] && iou(detections[idx].box, detections[other].box) > NMS_THRESHOLD) {
        suppressed[other] = true;
      }
    }
  }

  // ---------------- 5. 坐标逆映射 ----------------
  // Letterbox 变换：x_in = x_orig * ratio + pad.x
  // 逆变换：x_orig = (x_in - pad.x) / ratio
  const auto toOriginal = [ratio, pad](const cv::Point2f& p) {
    return cv::Point2f((p.x - pad.x) / ratio, (p.y - pad.y) / ratio);
  };

  // 训练标注的关键点顺序是 左上、左下、右下、右上（见 armor_pose.yaml 中
  // flip_idx: [3, 2, 1, 0]，水平翻转后顺序不变）。Armor 要求
  // 左上、右上、右下、左下，所以这里重排索引。
  // 实测不重排时平均角点误差从 6.5px 增大到 34.8px。
  constexpr int kCornerOrder[NUM_KEYPOINTS] = {0, 3, 2, 1};

  armors.reserve(keep.size());
  for (const int idx : keep) {
    const Detection& d = detections[idx];

    Armor armor;
    armor.confidence = d.confidence;
    armor.class_id = d.class_id;
    for (int k = 0; k < NUM_KEYPOINTS; ++k) {
      armor.corners[k] = toOriginal(d.kpts[kCornerOrder[k]]);
    }

    // 大小板判断：Pose 模型不直接输出该属性，这里按角点包围框的宽高比近似：
    // 小装甲板接近 1:1，大装甲板接近 2:1。阈值可按实际数据调整。
    const cv::Rect2f kpt_box = cv::boundingRect(
        std::vector<cv::Point2f>(armor.corners.begin(), armor.corners.end()));
    armor.large_armor = kpt_box.width > 0 && kpt_box.height > 0 &&
                        (kpt_box.width / kpt_box.height) > 1.5f;

    armors.push_back(armor);
  }

  // 按置信度降序返回，调用方可以直接取第一个作为最可信目标
  std::sort(armors.begin(), armors.end(),
            [](const Armor& a, const Armor& b) { return a.confidence > b.confidence; });

  // ---------------- 6. 可视化 ----------------
  // 检测框只画在本地副本上，input->image 保持原样：
  // 一是第四讲 PnP 要用不受绘制像素影响的图像，二是 main 里显示的画面由这里负责。
  if (DRAW_RESULT) {
    cv::Mat view = image.clone();
    drawArmors(view, armors);
    cv::imshow(kDetectionWindowName, view);
    WINDOW_CREATED = true;
  }
  return armors;
}
}  // namespace training
