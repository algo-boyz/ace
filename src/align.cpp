// align.cpp
#include "align.hpp"
#include <opencv2/calib3d.hpp>

static const std::array<cv::Point2f, 5> ARCFACE_DST = {{
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
}};

cv::Mat align_face(const cv::Mat& img, const FaceLandmarks& lm, int out_size) {
    std::vector<cv::Point2f> src(5), dst(5);
    for (int i = 0; i < 5; ++i) {
        src[i] = lm.pts[i];
        dst[i] = ARCFACE_DST[i] * (static_cast<float>(out_size) / 112.f);
    }

    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::LMEDS);
    if (M.empty()) {
        // fallback: simple resize of the box region
        return cv::Mat();
    }

    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(out_size, out_size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    return aligned;
}

FaceLandmarks approx_landmarks_from_box(const cv::Rect& box) {
    // Rough but usable approximation for frontal faces
    float x = box.x, y = box.y, w = box.width, h = box.height;
    FaceLandmarks lm;
    lm.pts[0] = {x + 0.30f * w, y + 0.35f * h}; // L eye
    lm.pts[1] = {x + 0.70f * w, y + 0.35f * h}; // R eye
    lm.pts[2] = {x + 0.50f * w, y + 0.55f * h}; // nose
    lm.pts[3] = {x + 0.35f * w, y + 0.78f * h}; // L mouth
    lm.pts[4] = {x + 0.65f * w, y + 0.78f * h}; // R mouth
    return lm;
}