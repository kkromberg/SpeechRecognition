/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "FeedForwardLayer.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

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

/*
 * H: Number of outputs
 * D: Feature size
 * B: Batch size
 */
void FeedForwardLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {

  /*
  std::cerr << "Batch   size B: " << batch_size_ << std::endl;
  std::cerr << "Feature size D: " << feature_size_ << std::endl;
  std::cerr << "Output  size H: " << output_size_  << std::endl;
  */

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

  // Go over every time step. Currently the mask is ignored due to simplification reasons.
  // Optimally, you would break after the max sequence length in the mask has been seen.
	for (size_t time_idx = 0; time_idx < max_seq_length_; time_idx++) {
	  if (time_idx > max_time_step) {
	    break;
	  }

	  // Slice of size B x D. This in the input to the layer
    std::gslice input_slice(time_idx  * batch_size_ * feature_size_,
                           {batch_size_, feature_size_},
                           {feature_size_, 1});

    std::valarray<float> input = input_buffer_[input_slice];

    // Slice of size B x H
    std::gslice output_slice(time_idx * slice.size()[1] * slice.size()[2],
                           {slice.size()[1], slice.size()[2]},
                           {slice.stride()[1], slice.stride()[2]});

    // prepare the results container
    std::valarray<float> result = bias;

/*
    std::cout << "Output dimensions: " << slice.size()[1] << " x " << slice.size()[2] << std::endl;
    std::cout << "Bias dimensions  : " << batch_size_ << " x " << output_size_ << std::endl;

    std::cerr << "Batch   size B: " << batch_size_ << std::endl;
    std::cerr << "Feature size D: " << feature_size_ << std::endl;
    std::cerr << "Output  size H: " << output_size_  << std::endl;
    std::cerr << "Size of bias   : " << bias.size() << std::endl;
    std::cerr << "Size of weights: " << weights.size() << std::endl;
    std::cerr << "Size of input  : " << input.size() << std::endl;

    std::cout << "Bias (Tranposed) : " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
       // std::cerr << std::endl  << batch_idx * output_size_ + output_idx << std::endl;
        std::cerr << bias[batch_idx * output_size_ + output_idx] << " " ;
      }
      std::cerr << std::endl;
    }

    std::cerr << "Weights (transposed) : " << std::endl;
    for (size_t feature_idx = 0; feature_idx < feature_size_; feature_idx++) {
      for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
        std::cerr << weights[output_idx * feature_size_ + feature_idx] << " " ;
      }
      std::cerr << std::endl;
    }

    std::cerr << "Input: " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t feature_idx = 0; feature_idx < feature_size_; feature_idx++) {
        std::cerr << input[feature_idx + batch_idx * feature_size_] << " " ;
      }
      std::cerr << std::endl;
    }

*/
    // Perform the following calculation:
    // output^(BxH) = input^(BxD) * weights^(DxH) + bias^(BxH)
    // The previously computed weights matrix has to be transposed
	  cblas_sgemm(CblasRowMajor,
	              CblasNoTrans,
	              CblasTrans,
	              batch_size_,
	              output_size_,
	              feature_size_,
	              1.0f,
	              &input[0],          // B x D
	              feature_size_,      // # columns in the input
	              &weights[0],        // D x H
	              feature_size_,      // # columns in the weights (untransposed)
	              1.0f,
	              &result[0],         // B x H <- this is the bias and the result at the same time
	              output_size_);      // # columns in the bias

    //applying different activation functions
    switch(nonlinearity_){
      case Nonlinearity::None: {
        output[output_slice] = result;
        break;
      }
      case Nonlinearity::Sigmoid: {
        output[output_slice] = 1/(1+exp(-result));
        break;
      }
      case Nonlinearity::Tanh: {
        output[output_slice] = 2/(1+exp(-2.0f*(result)))-1;
        break;
      }
      case Nonlinearity::ReLU:{
        output[output_slice] = result.apply([](float n)->float {
            if (n>0)
              return n;
            else
              return 0;
                  });
        break;
      }
    }
	}
}

void FeedForwardLayer::backward_start() {
  error_buffer_ = 0.0f;
  gradient_     = 0.0f;
}

void FeedForwardLayer::backward(std::valarray<float>& output, std::valarray<float>& error,
                                std::gslice const& slice, std::vector<unsigned> const& mask) {
  // TODO: implement
}

FeedForwardLayer::FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity)
                                  : NetworkLayer(config), nonlinearity_(nonlinearity) {
}

