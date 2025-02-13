#pragma once
#include <Eigen/Dense>
#include <filesystem>

namespace weight_utils {
    // Load 1D tensor (bias vectors)
    Eigen::RowVectorXf load_1d_tensor(const std::filesystem::path& path);

    // Load 2D tensor (weight matrices)
    Eigen::MatrixXf load_2d_tensor(const std::filesystem::path& path);

    // Verify tensor dimensions match expected shape
    void assert_tensor_shape(const Eigen::MatrixXf& tensor, int rows, int cols, std::string tensor_name = "[no_name]");
    void assert_vector_shape(const Eigen::RowVectorXf& vector, int size, std::string vector_name = "[no_name]");
}
