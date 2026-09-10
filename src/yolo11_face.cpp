#include "yolo11_face.hpp"
#include <iostream>
#include <algorithm>

Yolo11Face::Yolo11Face(const std::string& model_path, int input_size,
                       float conf_thres, float iou_thres)
    : env_(ORT_LOGGING_LEVEL_WARNING, "Yolo11Face"),
      input_size_(input_size), conf_thres_(conf_thres), iou_thres_(iou_thres)
{
    options_.SetIntraOpNumThreads(4);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_COREML
    uint32_t coreml_flags = 0;
    // coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options_, coreml_flags));
    std::cout << "[Yolo11Face] CoreML EP enabled\n";
#endif

    session_ = Ort::Session(env_, model_path.c_str(), options_);

    // Input / output names
    size_t num_inputs = session_.GetInputCount();
    size_t num_outputs = session_.GetOutputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session_.GetInputNameAllocated(i, allocator_);
        input_names_.push_back(name.get());
        input_names_cstr_.push_back(input_names_.back().c_str());
    }
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.push_back(name.get());
        output_names_cstr_.push_back(output_names_.back().c_str());
    }
}

cv::Mat Yolo11Face::letterbox(const cv::Mat& img, float& scale, int& pad_x, int& pad_y) {
    int w = img.cols, h = img.rows;
    scale = std::min(static_cast<float>(input_size_) / w,
                     static_cast<float>(input_size_) / h);
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    pad_x = (input_size_ - new_w) / 2;
    pad_y = (input_size_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::Mat out(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return out;
}

std::vector<Detection> Yolo11Face::detect(const cv::Mat& bgr) {
    float scale;
    int pad_x, pad_y;
    cv::Mat lb = letterbox(bgr, scale, pad_x, pad_y);

    // NCHW float32, 0-1
    cv::Mat blob;
    cv::dnn::blobFromImage(lb, blob, 1.0/255.0, cv::Size(), cv::Scalar(), true, false);

    std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_cstr_.data(), &input_tensor, 1,
                                output_names_cstr_.data(), output_names_cstr_.size());

    // Ultralytics YOLO11 output is typically [1, 4+nc, num_proposals] → transpose to [num, 4+nc]
    float* data = outputs[0].GetTensorMutableData<float>();
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    // shape is usually [1, 5, 8400] for single-class face (4 box + 1 score)
    int num_attrs = static_cast<int>(shape[1]);
    int num_proposals = static_cast<int>(shape[2]);

    std::vector<Detection> dets;
    postprocess(data, num_proposals, num_attrs, scale, pad_x, pad_y,
                bgr.cols, bgr.rows, dets);
    return dets;
}

void Yolo11Face::postprocess(const float* data, int num_proposals, int num_attrs,
                             float scale, int pad_x, int pad_y,
                             int orig_w, int orig_h, std::vector<Detection>& out)
{
    // data layout: [attr][proposal]  → we transpose on the fly
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < num_proposals; ++i) {
        float score = data[4 * num_proposals + i];   // class score (single class)
        if (score < conf_thres_) continue;

        float cx = data[0 * num_proposals + i];
        float cy = data[1 * num_proposals + i];
        float w  = data[2 * num_proposals + i];
        float h  = data[3 * num_proposals + i];

        float x1 = (cx - w/2.f - pad_x) / scale;
        float y1 = (cy - h/2.f - pad_y) / scale;
        float x2 = (cx + w/2.f - pad_x) / scale;
        float y2 = (cy + h/2.f - pad_y) / scale;

        x1 = std::clamp(x1, 0.f, static_cast<float>(orig_w - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(orig_h - 1));
        x2 = std::clamp(x2, 0.f, static_cast<float>(orig_w - 1));
        y2 = std::clamp(y2, 0.f, static_cast<float>(orig_h - 1));

        boxes.emplace_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
        scores.push_back(score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres_, iou_thres_, indices);

    for (int idx : indices) {
        Detection d;
        d.box = boxes[idx];
        d.score = scores[idx];
        d.landmarks = approx_landmarks_from_box(d.box);
        out.push_back(d);
    }
}