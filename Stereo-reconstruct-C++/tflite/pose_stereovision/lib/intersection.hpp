// intersection.hpp
#ifndef INTERSECTION_HPP
#define INTERSECTION_HPP

#include <vector>
#include <map>
#include <cmath> // For std::sqrt, std::abs, std::max, std::min, std::pow, etc.
#include <algorithm> // For std::max_element and other potential algorithm functions

// Declare the functions
float phi(float z);
float point_match(float dist1_mu, float dist1_sig, float val);
float dist_intersect(float dist1_mu, float dist1_sig, float dist2_mu, float dist2_sig);
float est_sig(float rng, float prob, const std::map<float, float>& map_probs);
std::map<float, float> gen_dict();
float intersect(const std::vector<float>& means1, const std::vector<float>& means2, const std::vector<float>& range1, const std::vector<float>& range2, float prob, const std::map<float, float>& map_probs);

#endif // INTERSECTION_HPP