#include <cmath>
#include <map>
#include <vector>
#include <iostream>
#include <algorithm>

double phi(double z) {
    return (1.0 + std::erf(z / std::sqrt(2.0))) / 2.0;
}

double point_match(double dist1_mu, double dist1_sig, double val) {
    double z = std::abs((val - dist1_mu) / dist1_sig);
    double cum_prob = phi(z);
    return 1 - cum_prob;
}

double dist_intersect(double dist1_mu, double dist1_sig, double dist2_mu, double dist2_sig) {
    double step_sig = std::max(dist1_sig, dist2_sig);
    double step = 6 * step_sig / 10;
    double startx = std::min(dist1_mu, dist2_mu) - 6 * step_sig;
    double endx = std::max(dist1_mu, dist2_mu) + 6 * step_sig;
    double int_prob = 0;

    for (double currx = startx; currx < endx; currx += step) {
        double refz1 = (currx - dist1_mu) / dist1_sig;
        double refz2 = ((currx + step) - dist1_mu) / dist1_sig;
        double p1 = phi(refz1);
        double p2 = phi(refz2);
        double prob1 = std::abs(p2 - p1);

        refz1 = (currx - dist2_mu) / dist2_sig;
        refz2 = ((currx + step) - dist2_mu) / dist2_sig;
        p1 = phi(refz1);
        p2 = phi(refz2);
        double prob2 = std::abs(p2 - p1);

        int_prob += std::min(prob1, prob2);
    }

    return int_prob;
}

double est_sig(double rng, double prob, const std::map<double, double>& map_probs) {
    prob = std::round(prob * 100) / 100;
    auto it = map_probs.find(prob);
    if (it != map_probs.end()) {
        return 0.5 * rng / it->second;
    }
    return (prob <= 0 || prob > 1) ? -1 : -1;
}

std::map<double, double> gen_dict() {
    std::map<double, double> map_probs;
    for (int x = 1; x < 350; ++x) {
        double prob = std::round((phi(x / 100.0) - phi(-x / 100.0)) * 100) / 100;
        if (map_probs.find(prob) == map_probs.end()) {
            map_probs[prob] = x / 100.0;
        }
    }
    return map_probs;
}

std::pair<double, std::vector<double>> intersect(const std::vector<double>& means1, const std::vector<double>& means2, const std::vector<double>& range1, const std::vector<double>& range2, double prob, const std::map<double, double>& map_probs) {
    double mult = 1;
    std::vector<double> probs;
    double tot = 0;
    int cnt = 0, nan_cnt = 0;

    for (size_t i = 0; i < means1.size(); ++i) {
        double sig1 = est_sig(range1[i], prob, map_probs);
        double sig2 = est_sig(range2[i], prob, map_probs);
        if (sig1 == -1 || sig2 == -1) {
            probs.push_back(std::nan(""));
            nan_cnt++;
            continue;
        }

        double int_prob = dist_intersect(means1[i], sig1, means2[i], sig2);
        if (int_prob == 0) {
            probs.push_back(std::nan(""));
            nan_cnt++;
            continue;
        }

        mult *= int_prob;
        tot += int_prob;
        cnt++;
        probs.push_back(int_prob);
    }

    double avg_prob = cnt > 0 ? tot / cnt : 0;
    if (nan_cnt > 0) {
        mult *= std::pow(10, -nan_cnt);
    }

    return {mult, probs};
}

std::pair<double, std::vector<double>> intersect_or(const std::vector<double>& means1, const std::vector<double>& means2, const std::vector<double>& range1, const std::vector<double>& range2, double prob, const std::map<double, double>& map_probs) {
    double mult = 1;
    std::vector<double> probs;
    double tot = 0;
    int cnt = 0, nan_cnt = 0;

    for (size_t i = 0; i < means1.size(); ++i) {
        double sig1 = est_sig(range1[i], prob, map_probs);
        double sig2 = est_sig(range2[i], prob, map_probs);
        if (sig1 == -1 || sig2 == -1) {
            probs.push_back(std::nan(""));
            nan_cnt++;
            continue;
        }

        double int_prob = dist_intersect(means1[i], sig1, means2[i], sig2);
        if (int_prob == 0) {
            probs.push_back(std::nan(""));
            nan_cnt++;
            continue;
        }

        mult += int_prob;
        tot += int_prob;
        cnt++;
        probs.push_back(int_prob);
    }

    mult = cnt > 0 ? mult / cnt : 0;

    return {mult, probs};
}



int main() {
    double ret_val = dist_intersect(1, 1, 6, 1);
    std::cout << "dist_intersect: " << ret_val << std::endl;

    std::map<double, double> map_probs = gen_dict();

    double distri = 0.677;
    std::vector<double> mean1 = {32.51783905029297, 22.714483642578124, 18.473861694335938, 23.965794372558594, 20.531011962890624, 36.12490539550781, 35.75204772949219, 34.6367431640625, 32.66388854980469, 48.96383972167969, 50.294253540039065, 20.608946228027342};
    std::vector<double> range1 = {1.5, 0.7120620727539055, 1.5, 0.5262313842773452, 0.833995056152343, 1.004507446289061, 0.8336090087890611, 1.1875183105468778, 1.5, 1.3794799804687514, 1.5, 0.5};

    std::vector<double> mean2 = {31.859530639648437, 24.62757110595703, 20.4177490234375, 22.398268127441405, 22.508877563476563, 36.436947631835935, 36.81101684570312, 37.32706909179687, 35.57789001464844, 45.0595718383789, 47.786416625976564, 19.50404052734375};
    std::vector<double> range2 = {0.5, 0.8721343994140653, 1.0946456909179716, 1.4546157836914055, 1.5, 0.5, 0.9813110351562528, 0.9918975830078125, 1.5, 1.5, 0.7932556152343722, 0.7743011474609389};

    auto [mult, probs] = intersect(mean1, mean2, range1, range2, distri, map_probs);
    std::cout << "intersect: " << mult << std::endl;
    for (double prob : probs) {
        std::cout << prob << " ";
    }
    std::cout << std::endl;

    return 0;
}


//to run this: g++ -o calc_match calc_match.cpp -ltensorflow-lite -lopencv_core -lopencv_imgproc -lopencv_highgui `pkg-config --cflags --libs opencv4` && ./test_unordered
