// arcface.hpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ArcFace {
public:
    ArcFace(const std::string& model_path);
    std::vector<float> embed(const cv::Mat& aligned_bgr_112);  // L2-normalized 512-d

private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::SessionOptions options_;
    std::vector<std::string> input_names_, output_names_;
    std::vector<const char*> input_names_cstr_, output_names_cstr_;
};