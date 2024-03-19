// intersection.cpp
#include "intersection.hpp"
#include <limits> // For NAN
#include <cmath> // Already included in the header, but reiterated here for clarity

// Definitions of the functions
float phi(float z) {
    return (1.0f + std::erf(z / std::sqrt(2.0f))) / 2.0f;
}

float point_match(float dist1_mu, float dist1_sig, float val) {
    float z = std::abs((val - dist1_mu) / dist1_sig);
    float cum_prob = phi(z);
    return 1.0f - cum_prob;
}

float dist_intersect(float dist1_mu, float dist1_sig, float dist2_mu, float dist2_sig) {
    float step_sig = std::max(dist1_sig, dist2_sig);
    float step = 6.0f * step_sig / 10.0f;
    float startx = std::min(dist1_mu, dist2_mu) - 6.0f * step_sig;
    float endx = std::max(dist1_mu, dist2_mu) + 6.0f * step_sig;
    float int_prob = 0.0f;

    for (float currx = startx; currx < endx; currx += step) {
        float refz1 = (currx - dist1_mu) / dist1_sig;
        float refz2 = ((currx + step) - dist1_mu) / dist1_sig;
        float p1 = phi(refz1);
        float p2 = phi(refz2);
        float prob1 = std::abs(p2 - p1);

        refz1 = (currx - dist2_mu) / dist2_sig;
        refz2 = ((currx + step) - dist2_mu) / dist2_sig;
        p1 = phi(refz1);
        p2 = phi(refz2);
        float prob2 = std::abs(p2 - p1);

        int_prob += std::min(prob1, prob2);
    }

    return int_prob;
}

float est_sig(float rng, float prob, const std::map<float, float>& map_probs) {
    prob = std::round(prob * 100.0f) / 100.0f;
    auto it = map_probs.find(prob);
    if (it != map_probs.end()) {
        return 0.5f * rng / it->second;
    }
    return (prob <= 0.0f || prob > 1.0f) ? -1.0f : -1.0f;
}

std::map<float, float> gen_dict() {
    std::map<float, float> map_probs;
    for (int x = 1; x < 350; ++x) {
        float prob = std::round((phi(static_cast<float>(x) / 100.0f) - phi(static_cast<float>(-x) / 100.0f)) * 100.0f) / 100.0f;
        if (map_probs.find(prob) == map_probs.end()) {
            map_probs[prob] = static_cast<float>(x) / 100.0f;
        }
    }
    return map_probs;
}

float intersect(const std::vector<float>& means1, const std::vector<float>& means2, const std::vector<float>& range1, const std::vector<float>& range2, float prob, const std::map<float, float>& map_probs) {
    float mult = 1.0f;
    std::vector<float> probs;
    float tot = 0.0f;
    int cnt = 0, nan_cnt = 0;

    for (size_t i = 0; i < means1.size(); ++i) {
        float sig1 = est_sig(range1[i], prob, map_probs);
        float sig2 = est_sig(range2[i], prob, map_probs);
        if (sig1 == -1.0f || sig2 == -1.0f) {
            probs.push_back(NAN);
            nan_cnt++;
            continue;
        }

        float int_prob = dist_intersect(means1[i], sig1, means2[i], sig2);
        if (int_prob == 0.0f) {
            probs.push_back(NAN);
            nan_cnt++;
            continue;
        }

        mult *= int_prob;
        tot += int_prob;
        cnt++;
        probs.push_back(int_prob);
    }

    float avg_prob = cnt > 0 ? tot / cnt : 0.0f;
    if (nan_cnt > 0) {
        mult *= std::pow(10.0f, -nan_cnt);
    }

    return mult;
}


// int main() {
//     double ret_val = dist_intersect(1, 1, 6, 1);
//     std::cout << "dist_intersect: " << ret_val << std::endl;

//     std::map<double, double> map_probs = gen_dict();

//     double distri = 0.677;
//     std::vector<double> mean1 = {32.51783905029297, 22.714483642578124, 18.473861694335938, 23.965794372558594, 20.531011962890624, 36.12490539550781, 35.75204772949219, 34.6367431640625, 32.66388854980469, 48.96383972167969, 50.294253540039065, 20.608946228027342};
//     std::vector<double> range1 = {1.5, 0.7120620727539055, 1.5, 0.5262313842773452, 0.833995056152343, 1.004507446289061, 0.8336090087890611, 1.1875183105468778, 1.5, 1.3794799804687514, 1.5, 0.5};

//     std::vector<double> mean2 = {31.859530639648437, 24.62757110595703, 20.4177490234375, 22.398268127441405, 22.508877563476563, 36.436947631835935, 36.81101684570312, 37.32706909179687, 35.57789001464844, 45.0595718383789, 47.786416625976564, 19.50404052734375};
//     std::vector<double> range2 = {0.5, 0.8721343994140653, 1.0946456909179716, 1.4546157836914055, 1.5, 0.5, 0.9813110351562528, 0.9918975830078125, 1.5, 1.5, 0.7932556152343722, 0.7743011474609389};

//     auto [mult, probs] = intersect(mean1, mean2, range1, range2, distri, map_probs);
//     std::cout << "intersect: " << mult << std::endl;
//     for (double prob : probs) {
//         std::cout << prob << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }