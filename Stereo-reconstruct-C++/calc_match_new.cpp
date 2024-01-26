#include <cmath>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>

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

std::pair<float, std::vector<float>> intersect(const std::vector<float>& means1, const std::vector<float>& means2, const std::vector<float>& range1, const std::vector<float>& range2, float prob, const std::map<float, float>& map_probs) {
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

    return {mult, probs};
}




int main() {
    float ret_val = dist_intersect(1.0f, 1.0f, 6.0f, 1.0f);
    std::cout << "dist_intersect: " << ret_val << std::endl;

    std::map<float, float> map_probs = gen_dict();

    float distri = 0.677f;
    std::vector<float> mean1 = {32.517839f, 22.714484f, 18.473862f, 23.965794f, 20.531012f, 36.124905f, 35.752048f, 34.636743f, 32.663889f, 48.963840f, 50.294254f, 20.608946f};
    std::vector<float> range1 = {1.5f, 0.712062f, 1.5f, 0.526231f, 0.833995f, 1.004507f, 0.833609f, 1.187518f, 1.5f, 1.379480f, 1.5f, 0.5f};

    std::vector<float> mean2 = {31.859531f, 24.627571f, 20.417749f, 22.398268f, 22.508878f, 36.436948f, 36.811017f, 37.327069f, 35.577890f, 45.059572f, 47.786417f, 19.504041f};
    std::vector<float> range2 = {0.5f, 0.872134f, 1.094646f, 1.454616f, 1.5f, 0.5f, 0.981311f, 0.991898f, 1.5f, 1.5f, 0.793256f, 0.774301f};

    auto [mult, probs] = intersect(mean1, mean2, range1, range2, distri, map_probs);
    std::cout << "intersect: " << mult << std::endl;
    for (float prob : probs) {
        std::cout << prob << " ";
    }
    std::cout << std::endl;

    return 0;
}



//to run this: g++ -o calc_match_new calc_match_new.cpp -ltensorflow-lite -lopencv_core -lopencv_imgproc -lopencv_highgui `pkg-config --cflags --libs opencv4` && ./calc_match_new
