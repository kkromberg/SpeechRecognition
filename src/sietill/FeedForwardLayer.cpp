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
  // TODO: implement
	int m = output_size_;
	int n = feature_size_;
	std::valarray<std::valarray<float>> W;
	std::valarray<float> x = input_buffer_[slice];
	std::valarray<float> temp=params_[std::slice(feature_size_*output_size_+1,output_size_,1)];
	cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, (float*)&W[0][0], m, &x[0], 1, 1.0f, &temp[0], 1);
	for(size_t i=0;i<output_size_;i++){
		W[i]=params_[std::slice(i*feature_size_,feature_size_,1)];
	}
	switch(nonlinearity_){
		case Nonlinearity::None: {
			output[slice] = temp;
		}
		case Nonlinearity::Sigmoid: {
			output[slice]=1/(1+exp(-temp));
		}
		case Nonlinearity::Tanh: {
			output[slice]=2/(1+exp(-2.0f*(temp)))-1;
		}
		case Nonlinearity::ReLU:{
			output[slice]=temp.apply([](float n)->float {
					if (n>0)
						return n;
					else
						return 0;
                });
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

