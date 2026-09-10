// align.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <array>

struct FaceLandmarks {
    std::array<cv::Point2f, 5> pts;   // L-eye, R-eye, nose, L-mouth, R-mouth
};

cv::Mat align_face(const cv::Mat& img, const FaceLandmarks& lm, int out_size = 112);

// Helper: approximate 5 landmarks from a detection box (good enough for demo)
FaceLandmarks approx_landmarks_from_box(const cv::Rect& box);