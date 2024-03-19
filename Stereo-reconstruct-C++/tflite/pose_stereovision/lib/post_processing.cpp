#include "post_processing.hpp"


//start of post processing mag algorithm
//****************************************
//********havent validated all
void printNestedVector(const std::vector<std::vector<float>>& nestedVector) {
    for (size_t i = 0; i < nestedVector.size(); ++i) {
        std::cout << "Inner vector " << i << ": ";
        for (size_t j = 0; j < nestedVector[i].size(); ++j) {
            std::cout << nestedVector[i][j] << " ";
        }
        std::cout << std::endl; // Print a newline at the end of each inner vector
    }
}


float calculateVariance(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f; // Handle empty vector case
    }

    // Step 1: Calculate the mean of non-zero values
    float sum = 0.0f;
    size_t count = 0;
    for (float value : data) {
        if (value != 0.0f) {
            sum += value;
            ++count;
        }
    }
    if (count == 0) {
        return 0.0f; // All values are zero or empty data
    }
    float mean = sum / count;

    // Step 2: Calculate the sum of squared differences from the mean for non-zero values
    float varianceSum = 0.0f;
    for (float value : data) {
        if (value != 0.0f) {
            varianceSum += (value - mean) * (value - mean);
        }
    }

    // Step 3: Calculate the variance
    float variance = varianceSum / count;
    return variance;
}



// class Frame {
// public:
//     std::vector<float> mean_local;
//     std::vector<float> range_local;
//     std::vector<std::vector<float>> parts_local;
//     std::string frame_name;

//     // Default constructor
//     Frame() = default;

//     // Existing constructor
//     Frame(const std::vector<float>& mean_local, const std::vector<float>& range_local, 
//           const std::vector<std::vector<float>>& parts_local, const std::string& frame_name)
//         : mean_local(mean_local), range_local(range_local), 
//           parts_local(parts_local), frame_name(frame_name) {}

//     void printMeanAndRange() const {
//         std::cout << "mean: ";
//         for (float value : mean_local) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;

//         std::cout << "range: ";
//         for (float value : range_local) {
//             std::cout << value << " ";
//         }
//         std::cout << std::endl;
//     }
//     bool isEmpty() const {
//         return mean_local.empty() && range_local.empty() && parts_local.empty() && frame_name.empty();
//     }

//     void printDetails() const {
//         // std::cout << "Frame Name: " << frame_name << std::endl;
//         printMeanAndRange();
//         // std::cout << "Parts Local:" << std::endl;
//         // for (const auto& part : parts_local) {
//         //     if(part.empty()){
//         //         std::cout << "empty";
//         //     }
//         //     else{
//         //         for (float value : part) {
//         //             std::cout << value << " ";
//         //         }
//         //     }

//         //     std::cout << std::endl;
//         // }
//     }


// };


Frame merge(const std::vector<std::vector<float>>& merge_parts, const std::string& string_a, const std::string& string_b) {
    std::vector<float> ranges;
    std::vector<float> means;
    std::vector<std::vector<float>> result_part;

    // std::cout << "-------------" << std::endl;
    // printNestedVector(merge_parts);
    
    for (const auto& part : merge_parts) {
        
        std::vector<float> temp = part;
        float range1 = 0.5f;
        float mean1 = 0.0f;

        if (part.empty()) {
            // continue; // Skip empty parts
            temp={};
            range1=0;
            mean1=0;

        }
 
        std::sort(temp.begin(), temp.end());
        size_t n = temp.size();


        size_t low_bound = static_cast<size_t>(0.4 * n);
        size_t up_bound = static_cast<size_t>(0.6 * n);

        if (n > 5) {


            // for (float value : temp) {
            //     std::cout << value << " ";
            // }
            // std::cout << std::endl;

            // assert (1==0);

            range1 = temp[up_bound] - temp[low_bound];
            mean1 = temp[n / 2]; // Median for odd number of elements
            if (n % 2 == 0) {
                mean1 = (temp[n / 2 - 1] + temp[n / 2]) / 2.0f; // Median for even number of elements
            }

            // std::cout << mean1 << range1 << std::endl;
            // assert (1==0);


        } else if (n == 5) {
            range1 = (temp[3] - temp[1]) / 2;
            mean1 = temp[2];
        } else if (n == 4) {
            range1 = (temp[2] - temp[1]) / 2;
            mean1 = (temp[1] + temp[2]) / 2;
        } else if (n == 3) {
            mean1 = temp[1];
        } else if (n == 2) {
            mean1 = (temp[0] + temp[1]) / 2;
        } else if ((n == 1)&&(temp[0]!=0)) {
            mean1 = temp[0];
        }

        if(range1!=0){
            range1 = std::max(0.5f, std::min(range1, 1.5f));
        }
        

        //assume no people segments could be ver 1 meters
        // assert ( mean1<100);
        if(mean1>100){
            temp={};
            range1=0;
            mean1=0;
        }



        result_part.push_back(temp);
        ranges.push_back(range1);
        means.push_back(mean1);
    }


    // for (float value : means) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;


    // for (float value : ranges) {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    // for (const auto& innerVec : result_part) {
    //     for (float value : innerVec) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << std::endl; // End of inner vector
    // }


    // assert (1==0);
    std::string merged_file_name;
    if(string_a==string_b){
        merged_file_name=string_a;
    }
    else{
        merged_file_name = string_a + string_b;
    }
    
    return Frame(means, ranges, result_part, merged_file_name);
}//merge have been validated


// Function to extract numerical part from filename and sort based on it
bool sortNumerically(const std::string& a, const std::string& b) {
    std::regex rgx("([0-9]+)");
    std::smatch matchA, matchB;

    if (std::regex_search(a, matchA, rgx) && std::regex_search(b, matchB, rgx)) {
        int numA = std::stoi(matchA[1]);
        int numB = std::stoi(matchB[1]);
        return numA < numB;
    }
    return a < b; // Fallback to lexicographical sorting
}


void printKeypoints(const std::vector<std::vector<Keypoint>>& keypoints) {
    for (size_t i = 0; i < keypoints.size(); ++i) {
        std::cout << "Vector " << i << ":" << std::endl;
        for (size_t j = 0; j < keypoints[i].size(); ++j) {
            std::cout << "Keypoint " << j << ": (" << keypoints[i][j].x << ", " << keypoints[i][j].y << ")" << std::endl;
        }
    }
}


std::vector<Keypoint> calculateAverageKeypoints(const std::vector<std::vector<Keypoint>>& keypoints) {
    if (keypoints.empty()) {
        return {}; // Return empty vector if input is empty
    }

    size_t numKeypoints = keypoints[0].size(); // Number of keypoints in each vector
    std::vector<Keypoint> averages(numKeypoints, {0, 0}); // Initialize averages with zero
    std::vector<int> validCounts(numKeypoints, 0); // Count of valid (non-zero) keypoints

    for (const auto& vec : keypoints) {
        if (vec.size() != numKeypoints) {
            std::cerr << "Inconsistent number of keypoints in vectors" << std::endl;
            return {}; // Handle error or inconsistency
        }
        for (size_t i = 0; i < numKeypoints; ++i) {
            if (vec[i].x != 0 && vec[i].y != 0) {
                averages[i].x += vec[i].x;
                averages[i].y += vec[i].y;
                validCounts[i]++;
            }
        }
    }

    for (size_t i = 0; i < numKeypoints; ++i) {
        if (validCounts[i] > 0) {
            averages[i].x /= validCounts[i];
            averages[i].y /= validCounts[i];
        }
    }

    return averages;
}


std::vector<float> findLargestBBox(const std::vector<std::vector<float>>& bboxes) {
    std::vector<float> largestBBox;
    float maxArea = 0.0f;

    for (const auto& bbox : bboxes) {
        // Ensure the class is 0
        if (bbox[5] == 0) {
            float width = bbox[2] - bbox[0];
            float height = bbox[3] - bbox[1];
            float area = width * height;

            if (area > maxArea) {
                maxArea = area;
                largestBBox = bbox;
            }
        }
    }

    return largestBBox;
}




void printListOfFrames(const std::vector<Frame>& frames) {
    for (const auto& frame : frames) {
        frame.printDetails();
        std::cout << "---------------------" << std::endl;
    }
}





//several function are not validated
// Function to merge two lists of floats
std::vector<std::vector<float>> merge_lists_of_lists(const std::vector<std::vector<float>>& list1, const std::vector<std::vector<float>>& list2) {
    if (list1.size() != list2.size()) {
        throw std::invalid_argument("Lists must be of the same length");
    }

    std::vector<std::vector<float>> merged_list;
    for (size_t i = 0; i < list1.size(); ++i) {
        // Merge inner vectors
        if (list1[i].empty() && list2[i].empty()) {
            merged_list.push_back({}); // Both empty, add empty vector
        } else if (list1[i].empty()) {
            merged_list.push_back(list2[i]); // Only list1 is empty, use list2
        } else if (list2[i].empty()) {
            merged_list.push_back(list1[i]); // Only list2 is empty, use list1
        } else {
            std::vector<float> merged_inner_list;
            merged_inner_list.insert(merged_inner_list.end(), list1[i].begin(), list1[i].end());
            merged_inner_list.insert(merged_inner_list.end(), list2[i].begin(), list2[i].end());
            merged_list.push_back(merged_inner_list);
        }
    }
    return merged_list;
}

std::vector<Frame> remove_duplicates(const std::vector<Frame>& frames) {
    std::vector<Frame> unique_frames;

    for (const auto& frame : frames) {
        // Check if a frame with the same frame_name is already in unique_frames
        auto it = std::find_if(unique_frames.begin(), unique_frames.end(),
                               [&frame](const Frame& unique_frame) {
                                   return unique_frame.frame_name == frame.frame_name;
                               });
        if (it == unique_frames.end()) {
            unique_frames.push_back(frame);
        }
    }
    return unique_frames;
}


std::vector<Frame> add_frame(std::vector<Frame>& local_database, const Frame& frame, double threshold, double distri,std::map<float, float> map_probs) {
    // Assuming remove_duplicates is a function that removes duplicate frames from the database
    local_database = remove_duplicates(local_database);

    if (local_database.empty()) {
        std::cout <<"add a new one to database" << std::endl;
        local_database.push_back(frame);
    } else {
        const auto& means_b = frame.mean_local;
        const auto& ranges_b = frame.range_local;
        bool found = false;

        for (size_t i = 0; i < local_database.size(); ++i) {
            const auto& means_a = local_database[i].mean_local;
            const auto& ranges_a = local_database[i].range_local;

            if (intersect(means_a, means_b, ranges_a, ranges_b, distri,  map_probs) > threshold) {
                auto merge_parts = merge_lists_of_lists(local_database[i].parts_local, frame.parts_local);

                std::cout << "found same one and merged!" << std::endl;
                Frame frame_final = merge(merge_parts, frame.frame_name, local_database[i].frame_name);
                local_database[i] = frame_final;
                found = true;
                break;
            }
        }

        if (!found) {
            std::cout << "add a new person" << std::endl;
            local_database.push_back(frame);
        }
    }

    local_database = remove_duplicates(local_database);
    return local_database;
}


std::vector<Frame> process_frames(const std::vector<Frame>& valid_frames, const std::vector<std::pair<size_t, size_t>>& pair_index, float threshold, int boundary_threshold, double distri, std::map<float, float> map_probs ) {
    std::vector<Frame> local_database;

    for (const auto& pair : pair_index) {
        if (intersect(valid_frames[pair.first].mean_local, valid_frames[pair.second].mean_local,
                      valid_frames[pair.first].range_local, valid_frames[pair.second].range_local, distri, map_probs) > threshold) {



            auto merged_parts = merge_lists_of_lists(valid_frames[pair.first].parts_local, valid_frames[pair.second].parts_local);

     
            Frame merged_frame = merge(merged_parts, valid_frames[pair.first].frame_name, valid_frames[pair.second].frame_name);

            Frame left_merged = merged_frame;
            Frame right_merged = merged_frame;
            int count = 0;

            for (int k = 1; k <= boundary_threshold; ++k) {
                if ((pair.first >= k) && intersect(merged_frame.mean_local, valid_frames[pair.first - k].mean_local, 
                        merged_frame.range_local, valid_frames[pair.first - k].range_local, distri, map_probs) > threshold) {
                
                    // std::cout<<"get left" << std::endl;
                    left_merged = merge(merge_lists_of_lists(left_merged.parts_local, valid_frames[pair.first - k].parts_local), 
                                        left_merged.frame_name, valid_frames[pair.first - k].frame_name);
                    count++;
                }

                if ((pair.second + k < valid_frames.size()) && intersect(merged_frame.mean_local, valid_frames[pair.second + k].mean_local, 
                        merged_frame.range_local, valid_frames[pair.second + k].range_local, distri, map_probs) > threshold) {

                    // std::cout<<"get right" << std::endl;

                    right_merged = merge(merge_lists_of_lists(right_merged.parts_local, valid_frames[pair.second + k].parts_local), 
                                         right_merged.frame_name, valid_frames[pair.second + k].frame_name);
                    count++;
                }
            }

            Frame final_frame = merge(merge_lists_of_lists(left_merged.parts_local, right_merged.parts_local), 
                                      left_merged.frame_name, right_merged.frame_name);
            // std::cout << "------by here is all correct." << std::endl;

            // final_frame.printDetails();
            // assert (0==1);

            local_database = add_frame(local_database, final_frame, threshold, distri, map_probs);
        }
        // Remove duplicates from the local database
        local_database = remove_duplicates(local_database);

        // std::cout << "At this pair the database is----------" << std::endl;
        // // Print the contents of local_database of each loop
        // for (const auto& frame : local_database) {
        //     frame.printDetails();
            
        // }
        // std::cout << "------------------" << std::endl;

        // assert (1==0);

    }



    // Optional: Additional logic for further processing or printing details

    return local_database;
}

//****************************************
//end of post processing mag algorithm


