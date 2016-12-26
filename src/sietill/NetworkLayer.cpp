/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "NetworkLayer.hpp"

#include <fstream>
#include <iostream>

#include "Util.hpp"

const ParameterString NetworkLayer::paramLayerName ("layer-name", "");
const ParameterUInt   NetworkLayer::paramOutputSize("num-outputs", 0u);

void NetworkLayer::init(bool input_error_needed) {
  input_error_needed_ = input_error_needed;
}

void NetworkLayer::save(std::string const& path) const {
  create_dir(path);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&params_[0]), params_.size() * sizeof(float));
}

void NetworkLayer::load(std::string const& path) {
  std::ifstream in(path, std::ios::in);
  if (not in.good()) {
    std::cerr << "error loading parameters: " << path << std::endl;
    std::abort();
  }
  in.read(reinterpret_cast<char*>(&params_[0]), params_.size() * sizeof(float));
}

void NetworkLayer::gradient_test() {
	size_t orig_batch_size = batch_size_;
	if (batch_size_ != 1) {
		std::cerr << "Gradient tests only implemented for batch size 1. Running with batch size 1" << std::endl;
		batch_size_ = 1;
	}

	float machine_eps = sqrt(1.19e-07);
	std::valarray<float> original_params = params_;

	// prepare containers for forward / backward passes
	std::gslice output_slice(0, {max_seq_length_, batch_size_, output_size_}, {batch_size_, output_size_, 1});
	std::valarray<float> error(0.0f, max_seq_length_ * output_size_  * batch_size_);
	std::vector<unsigned> mask(batch_size_, 0);
	mask[0] = 1;
	std::valarray<float>  input  (0.0f, max_seq_length_ * feature_size_ * batch_size_);

	// get a random input vector
	std::normal_distribution<float> dist(0.0f, 1.0f);
	std::mt19937 rng_device;
	for (size_t i = 0; i < feature_size_ * batch_size_; i++) {
		input[i] = 2.0f; //dist(rng_device);
	}

	for (size_t feature_idx = 0; feature_idx < feature_size_ + 1; feature_idx++) {
		for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {

			// get the parameter index
			size_t param_index = feature_idx + output_idx * (feature_size_);
			if (feature_idx == feature_size_) {
				param_index = feature_size_ * output_size_ + output_idx;
			}

			// set the error for the backprop to 1
			error = 0.0f;
			error[output_idx] = 1.0f;

			// forward pass with a slightly higher parameter
			std::valarray<float> output_plus (0.0f, max_seq_length_ * output_size_ * batch_size_);
			std::valarray<float> params_plus = params_;
			params_plus[param_index]  += machine_eps;
			params_       = params_plus;
			input_buffer_ = input;
			forward(output_plus, output_slice, mask);
			params_       = original_params;

			// forward pass with a slightly smaller parameter
			std::valarray<float> output_minus (0.0f, max_seq_length_ * output_size_ * batch_size_);
			std::valarray<float> params_minus = params_;
			params_minus[param_index] -= machine_eps;
			params_       = params_minus;
			input_buffer_ = input;
			forward(output_minus, output_slice, mask);

			// do backprop for gradient estimation
			std::valarray<float>  output (0.0f, max_seq_length_ * output_size_  * batch_size_);
			input_buffer_ = input;
			params_       = original_params;
			forward(output, output_slice, mask);
			backward_start();
			backward(output, error, output_slice, mask);

			// calculate approximative derivate
			float forward_gradient  = (output_plus[output_idx] - output_minus[output_idx]) / (2 * machine_eps);
			float backward_gradient = gradient_[param_index];

			// TODO: make gradient checks for the output layer
			if (fabs(backward_gradient - forward_gradient) > 1e-2 && get_layer_name() != "output-layer") {
				std::cerr << "Gradient test failed in layer: " << get_layer_name()  << " "
																											 << backward_gradient << " "
																											 << forward_gradient << std::endl;
			}
		}
	}

	batch_size_ = orig_batch_size;
}
