#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <filesystem>

namespace weight_utils {
    // Load 1D tensor (bias vectors)
    Eigen::VectorXf load_1d_tensor(const std::filesystem::path& path);

    // Load 2D tensor (weight matrices)
    Eigen::MatrixXf load_2d_tensor(const std::filesystem::path& path);

    // Verify tensor dimensions match expected shape
    bool verify_tensor_shape(const Eigen::MatrixXf& tensor, int rows, int cols);
    bool verify_tensor_shape(const Eigen::VectorXf& tensor, int size);
}
