/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "FeedForwardLayer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

#include <assert.h>
#include <cblas.h>

#include "Util.hpp"

namespace {
  FeedForwardLayer::Nonlinearity nonlinearity_from_string(std::string const& str) {
    if (str == "sigmoid") {
      return FeedForwardLayer::Nonlinearity::Sigmoid;
    }
    else if (str == "tanh") {
      return FeedForwardLayer::Nonlinearity::Tanh;
    }
    else if (str == "relu") {
      return FeedForwardLayer::Nonlinearity::ReLU;
    }
    return FeedForwardLayer::Nonlinearity::None;
  }

  double sigmoid(double x) {
	  return 1 / (1 + exp(-x));
  }

  double tanh(double x) {
	  return 2 * sigmoid(2*x) - 1;
  }

  double relu(double x) {
	  return std::max(0.0, x);
  }
}

const ParameterString FeedForwardLayer::paramNonlinearity("nonlinearity", "");

FeedForwardLayer::FeedForwardLayer(Configuration const& config) : NetworkLayer(config),
                                                                  nonlinearity_(nonlinearity_from_string(paramNonlinearity(config))) {
}

FeedForwardLayer::~FeedForwardLayer() {
}

void FeedForwardLayer::init(bool input_error_needed) {
  NetworkLayer::init(input_error_needed);
  params_.resize(feature_size_ * output_size_ + output_size_);
  gradient_.resize(params_.size());
}

void FeedForwardLayer::init_parameters(std::function<float()> const& generator) {
  for (size_t i = 0ul; i < params_.size(); i++) {
    params_[i] = generator();
  }
}

void FeedForwardLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  // TODO: implement
}

void FeedForwardLayer::backward_start() {
  error_buffer_ = 0.0f;
  gradient_     = 0.0f;
}

void FeedForwardLayer::backward(std::valarray<float>& output, std::valarray<float>& error,
                                std::gslice const& slice, std::vector<unsigned> const& mask) {
  // TODO: implement
  unsigned max_time_step = *std::max_element(mask.begin(), mask.end());

  // Matrix of size H x D
  std::gslice weights_slice(0, {output_size_, feature_size_}, {feature_size_, 1});
  std::valarray<float> weights = params_[weights_slice];

  // The bias is a matrix B x H. Each row of the matrix is the same!
  std::valarray<float> bias(0.0, output_size_ * batch_size_);
  for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
    for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
      bias[batch_idx * output_size_ + output_idx] = params_[output_size_ * feature_size_ + output_idx];
    }
  }

  // Go over every time step
	for (size_t time_idx = 0; time_idx < max_seq_length_; time_idx++) {
	  if (time_idx > max_time_step) {
	    break;
	  }
    // Slice of size B x H
    std::gslice output_slice(time_idx * slice.size()[1] * slice.size()[2],
                           {slice.size()[1], slice.size()[2]},
                           {slice.stride()[1], slice.stride()[2]});

    // dE_{total}/dw_{ij} = dE_{total}/dout_{ij} * dout{ij}/dnet_{ij} * dnet_{ij}/dw_{ij}
    // 1: dE_{total}/dout_{ij} sum partial derivations w.r.t. the output
    //std::valarray<float> sums_output_gradients = ??


	  // 2: dout{ij}/dnet_{ij} how much does the output changes w.r.t. the input
	  std::valarray<float> inner_gradients;
		switch (nonlinearity_) {
			case Nonlinearity::None:
				inner_gradients = 1.0f;
				break;
			case Nonlinearity::Sigmoid:
				//inner_gradients = sigmoid(0.0) * (1 - sigmoid(0.0));
				inner_gradients = output[output_slice] * (1- output[output_slice]);
				break;
			case Nonlinearity::Tanh:
				//inner_gradients = 1 - tanh(0.0) * tanh(0.0);
				inner_gradients = 1 - (output[output_slice] * output[output_slice]);
				break;
			case Nonlinearity::ReLU:
				float x = 0;
				if (output[output_slice] > 0.0f) {
					inner_gradients = 1.0f;
				}
				else {
					inner_gradients = 0.0f;
				}
				break;
			default:
				break;
		}


	  // 3: dnet_{ij}/dw_{ij} how much does the input changes w.r.t. to the weights
		//std::valarray<float> input_gradients = ??
		// store gradients
		//gradient_[slice] = sums_output_gradients * inner_gradients * input_gradients
	}

	// compute error if need
	if (input_error_needed_) {
		// TODO compute error and store in error_buffer_
	}



}

FeedForwardLayer::FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity)
                                  : NetworkLayer(config), nonlinearity_(nonlinearity) {
}

