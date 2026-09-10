#include "arcface.hpp"
#include <cmath>
#include <iostream>

ArcFace::ArcFace(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ArcFace")
{
    options_.SetIntraOpNumThreads(2);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_COREML
    std::unordered_map<std::string, std::string> provider_options;
    options_.AppendExecutionProvider("CoreML", provider_options);
    std::cout << "[ArcFace] CoreML EP enabled\n";
#endif

    session_ = Ort::Session(env_, model_path.c_str(), options_);

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = session_.GetInputNameAllocated(0, allocator);
    input_names_.push_back(in_name.get());
    input_names_cstr_.push_back(input_names_.back().c_str());

    auto out_name = session_.GetOutputNameAllocated(0, allocator);
    output_names_.push_back(out_name.get());
    output_names_cstr_.push_back(output_names_.back().c_str());
}

std::vector<float> ArcFace::embed(const cv::Mat& aligned) {
    // BGR → RGB, normalize to [-1, 1]
    cv::Mat rgb;
    cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0/128.0, -127.5/128.0);   // now HWC float

    // Keep HWC layout (no channel split)
    std::vector<float> input(112 * 112 * 3);
    std::memcpy(input.data(), rgb.data, input.size() * sizeof(float));

    // NHWC shape that the model expects
    std::array<int64_t, 4> shape{1, 112, 112, 3};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_cstr_.data(), &tensor, 1,
                                output_names_cstr_.data(), 1);

    float* emb = outputs[0].GetTensorMutableData<float>();
    std::vector<float> result(emb, emb + 512);

    // L2 normalize
    float norm = 0.f;
    for (float v : result) norm += v * v;
    norm = std::sqrt(norm) + 1e-8f;
    for (float& v : result) v /= norm;

    return result;
}