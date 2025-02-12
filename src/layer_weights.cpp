#include "layer_weights.hpp"

Linear::Linear(Eigen::MatrixXf weight, Eigen::VectorXf _bias) : weight(weight) {
    bias = _bias.transpose();
}

Linear::Linear() {}