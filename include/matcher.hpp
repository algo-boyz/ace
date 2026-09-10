// matcher.hpp
#pragma once
#include "hnswlib/hnswlib.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

class FaceMatcher {
public:
    FaceMatcher(int dim = 512, int max_elements = 10000);

    void add(const std::string& id, const std::vector<float>& emb);
    std::vector<std::pair<std::string, float>> search(const std::vector<float>& query, int k = 5);

    static float cosine(const std::vector<float>& a, const std::vector<float>& b);

private:
    hnswlib::InnerProductSpace space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::unordered_map<hnswlib::labeltype, std::string> id_map_;
    hnswlib::labeltype next_id_ = 0;
};