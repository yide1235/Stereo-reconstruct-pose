// post_processing.hpp
#ifndef POST_PROCESSING_HPP
#define POST_PROCESSING_HPP

#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <map>
#include "common_types.hpp" // If you use Keypoint in this header
#include "intersection.hpp"

using namespace std;





class Frame {
public:
    std::vector<float> mean_local;
    std::vector<float> range_local;
    std::vector<std::vector<float>> parts_local;
    std::string frame_name;

    // Default constructor
    Frame() = default;

    // Existing constructor
    Frame(const std::vector<float>& mean_local, const std::vector<float>& range_local, 
          const std::vector<std::vector<float>>& parts_local, const std::string& frame_name)
        : mean_local(mean_local), range_local(range_local), 
          parts_local(parts_local), frame_name(frame_name) {}

    void printMeanAndRange() const {
        std::cout << "mean: ";
        for (float value : mean_local) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        std::cout << "range: ";
        for (float value : range_local) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    bool isEmpty() const {
        return mean_local.empty() && range_local.empty() && parts_local.empty() && frame_name.empty();
    }

    void printDetails() const {
        // std::cout << "Frame Name: " << frame_name << std::endl;
        printMeanAndRange();
        // std::cout << "Parts Local:" << std::endl;
        // for (const auto& part : parts_local) {
        //     if(part.empty()){
        //         std::cout << "empty";
        //     }
        //     else{
        //         for (float value : part) {
        //             std::cout << value << " ";
        //         }
        //     }

        //     std::cout << std::endl;
        // }
    }


};



// Function declarations
void printNestedVector(const std::vector<std::vector<float>>& nestedVector);
float calculateVariance(const std::vector<float>& data);
bool sortNumerically(const std::string& a, const std::string& b);
void printKeypoints(const std::vector<std::vector<Keypoint>>& keypoints);
std::vector<Keypoint> calculateAverageKeypoints(const std::vector<std::vector<Keypoint>>& keypoints);
std::vector<float> findLargestBBox(const std::vector<std::vector<float>>& bboxes);
void printListOfFrames(const std::vector<Frame>& frames);
std::vector<std::vector<float>> merge_lists_of_lists(const std::vector<std::vector<float>>& list1, const std::vector<std::vector<float>>& list2);
std::vector<Frame> remove_duplicates(const std::vector<Frame>& frames);
std::vector<Frame> add_frame(std::vector<Frame>& local_database, const Frame& frame, double threshold, double distri,std::map<float, float> map_probs);
std::vector<Frame> process_frames(const std::vector<Frame>& valid_frames, const std::vector<std::pair<size_t, size_t>>& pair_index, float threshold, int boundary_threshold, double distri, std::map<float, float> map_probs);
Frame merge(const std::vector<std::vector<float>>& merge_parts, const std::string& string_a, const std::string& string_b);

#endif // POST_PROCESSING_HPP

