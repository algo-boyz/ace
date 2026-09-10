// matcher.cpp
#include "matcher.hpp"
#include <cmath>

FaceMatcher::FaceMatcher(int dim, int max_elements)
    : space_(dim)
{
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        &space_, max_elements, 16, 200);
}

void FaceMatcher::add(const std::string& id, const std::vector<float>& emb) {
    index_->addPoint(emb.data(), next_id_);
    id_map_[next_id_] = id;
    ++next_id_;
}

std::vector<std::pair<std::string, float>> FaceMatcher::search(
    const std::vector<float>& query, int k)
{
    auto result = index_->searchKnn(query.data(), k);
    std::vector<std::pair<std::string, float>> out;
    while (!result.empty()) {
        auto [dist, label] = result.top();
        result.pop();
        // InnerProduct space → higher = more similar (already L2-normed → cosine)
        out.emplace_back(id_map_[label], dist);
    }
    return out;
}

float FaceMatcher::cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-8f);
}