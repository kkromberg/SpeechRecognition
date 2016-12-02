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

void FeedForwardLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
	std::slice weights_slice(0, {output_size_, feature_size_}, {feature_size_, 1});
  std::slice bias_slice(output_size_ * feature_size_, output_size_, 1);

  float *weights = params_[weights_slice];
  //float *bias    = params_[bias_slice];

  // The bias is a matrix H x B, where H is the output_size_ and B is the batch_size_. Each column of the matrix is the same!
  // Find a way of initializing it correctly. (And then make a separate variable, because the matrix operations all get written to this matrix)
  float bias[output_size_ * batch_size_];

	for (size_t time_idx = 0; time_idx < max_seq_length_; time_idx++) {

    std::slice input_slice(time_idx * max_seq_length_,
                           {batch_size_, feature_size_},
                           {feature_size_, 1});

    std::slice output_slice(time_idx * slice.stride()[0],
                           {slice.size()[1], slice.size()[2]},
                           {slice.stride()[1], slice.stride()[2]});

    float *input = input_buffer_[input_slice];

	  cblas_dgemm(CblasRowMajor,
	              CblasNoTrans,
	              CblasNoTrans,
	              output_size_,
	              feature_size_,
	              batch_size_,
	              1.0f,
	              weights,
	              output_size_,
	              input,
	              feature_size_,
	              1.0f,
	              bias,
	              output_size_);

    //applying different activation functions
    switch(nonlinearity_){
      case Nonlinearity::None: {
        output[output_slice] = bias;
        break;
      }
      case Nonlinearity::Sigmoid: {
        output[output_slice] = 1/(1+exp(-bias));
        break;
      }
      case Nonlinearity::Tanh: {
        output[output_slice] = 2/(1+exp(-2.0f*(bias)))-1;
        break;
      }
      case Nonlinearity::ReLU:{
        output[output_slice] = bias.apply([](float n)->float {
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

