#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "align.hpp"
#include <vector>
#include <string>

struct Detection {
    cv::Rect box;
    float score;
    FaceLandmarks landmarks;   // filled by approx for now
};

class Yolo11Face {
public:
    Yolo11Face(const std::string& model_path, int input_size = 640,
               float conf_thres = 0.45f, float iou_thres = 0.45f);
    std::vector<Detection> detect(const cv::Mat& bgr);

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    int input_size_;
    float conf_thres_, iou_thres_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<const char*> input_names_cstr_, output_names_cstr_;

    cv::Mat letterbox(const cv::Mat& img, float& scale, int& pad_x, int& pad_y);
    void postprocess(const float* data, int num_proposals, int num_attrs,
                     float scale, int pad_x, int pad_y,
                     int orig_w, int orig_h, std::vector<Detection>& out);
};