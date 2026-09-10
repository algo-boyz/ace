#include "yolo11_face.hpp"
#include "arcface.hpp"
#include "align.hpp"
#include "matcher.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::string det_model = "models/yolo11n-face.onnx";
    std::string emb_model = "models/arcface_r50.onnx";
    std::string source = "0";          // webcam by default
    if (argc > 1) source = argv[1];    // can be path to mp4

    Yolo11Face detector(det_model);
    ArcFace    embedder(emb_model);
    FaceMatcher matcher(512);

    // Optional: pre-load a gallery
    // matcher.add("alice", some_embedding);

    cv::VideoCapture cap;
    if (source == "0" || source == "webcam")
        cap.open(0);
    else
        cap.open(source);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open source: " << source << "\n";
        return 1;
    }

    cv::Mat frame;
    auto t0 = std::chrono::steady_clock::now();
    int frames = 0;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        auto dets = detector.detect(frame);

        for (const auto& d : dets) {
            cv::rectangle(frame, d.box, {0, 255, 0}, 2);
            cv::putText(frame, cv::format("%.2f", d.score),
                        {d.box.x, d.box.y - 5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1);

            cv::Mat aligned = align_face(frame, d.landmarks);
            if (aligned.empty()) continue;

            auto emb = embedder.embed(aligned);

            // 1:N search
            auto hits = matcher.search(emb, 3);
            for (const auto& [id, score] : hits) {
                if (score > 0.45f) {   // typical ArcFace threshold
                    cv::putText(frame, id + " " + cv::format("%.2f", score),
                                {d.box.x, d.box.y + d.box.height + 15},
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 255}, 1);
                }
            }
        }

        // FPS
        ++frames;
        auto t1 = std::chrono::steady_clock::now();
        double fps = frames / std::chrono::duration<double>(t1 - t0).count();
        cv::putText(frame, cv::format("FPS: %.1f", fps), {10, 30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 255, 255}, 2);

        cv::imshow("YOLOv11 Face Recognition", frame);
        if (cv::waitKey(1) == 27) break;   // ESC
    }
    return 0;
}