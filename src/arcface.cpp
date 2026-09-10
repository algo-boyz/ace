#include "arcface.hpp"
#include <cmath>
#include <iostream>

ArcFace::ArcFace(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ArcFace")
{
    options_.SetIntraOpNumThreads(2);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_COREML
    uint32_t coreml_flags = 0;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options_, coreml_flags));
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
    rgb.convertTo(rgb, CV_32F, 1.0/128.0, -127.5/128.0);

    // HWC → CHW
    std::vector<float> input(3 * 112 * 112);
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c)
        std::memcpy(input.data() + c * 112 * 112, channels[c].data, 112 * 112 * sizeof(float));

    std::array<int64_t, 4> shape{1, 3, 112, 112};
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