#include "weight_utils.hpp"
#include <cnpy.h>

namespace weight_utils {
    Eigen::RowVectorXf load_1d_tensor(const std::filesystem::path& path) {
        cnpy::NpyArray arr = cnpy::npy_load(path.string());
        return Eigen::Map<Eigen::RowVectorXf>(arr.data<float>(), arr.shape[0]);
    }

    Eigen::MatrixXf load_2d_tensor(const std::filesystem::path& path) {
        cnpy::NpyArray arr = cnpy::npy_load(path.string());
        return Eigen::Map<Eigen::MatrixXf>(
            arr.data<float>(),
            arr.shape[0],
            arr.shape[1]
        );
    }

    void assert_tensor_shape(const Eigen::MatrixXf& tensor, int rows, int cols, std::string tensor_name) {
        if (tensor.rows() != rows || tensor.cols() != cols) {
            std::ostringstream oss;
            oss << "Tensor shape mismatch for '" << tensor_name << "'.\n";
            oss << "Expected dimensions [" << rows << ", " << cols << "]\n";
            oss << "Got dimensions [" << tensor.rows() << ", " << tensor.cols() << "]\n";
            throw std::runtime_error(oss.str());
        }
    }

    void assert_vector_shape(const Eigen::RowVectorXf& vector, int size, std::string vector_name) {
        if (vector.size() != size) {
            std::ostringstream oss;
            oss << "vector shape mismatch for '" << vector_name << "'.\n";
            oss << "Expected dimensions [" << size << "]\n";
            oss << "Got dimensions [" << vector.size() << "]\n";
            throw std::runtime_error(oss.str());
        }
    }
}
