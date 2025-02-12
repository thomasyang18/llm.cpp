// src/weight_utils.cpp
#include "weight_utils.hpp"
#include <cnpy.h>
#include <filesystem>

namespace weight_utils {
    Eigen::VectorXf load_1d_tensor(const std::string& path) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        return Eigen::Map<Eigen::VectorXf>(arr.data<float>(), arr.shape[0]);
    }
    
    Eigen::MatrixXf load_2d_tensor(const std::string& path) {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        return Eigen::Map<Eigen::MatrixXf>(
            arr.data<float>(), 
            arr.shape[0], 
            arr.shape[1]
        );
    }
    
    bool verify_tensor_shape(const Eigen::MatrixXf& tensor, int rows, int cols) {
        return tensor.rows() == rows && tensor.cols() == cols;
    }
    
    bool verify_tensor_shape(const Eigen::VectorXf& tensor, int size) {
        return tensor.size() == size;
    }
}
