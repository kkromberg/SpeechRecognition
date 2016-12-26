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

  FeedForwardLayer::WeightDecay weight_decay_from_string(std::string const& str) {
  	if (str == "l1") {
      return FeedForwardLayer::WeightDecay::L1;
    }
    else if (str == "l2") {
      return FeedForwardLayer::WeightDecay::L2;
    }
    return FeedForwardLayer::WeightDecay::None;
  }
}

const ParameterString FeedForwardLayer::paramNonlinearity("nonlinearity", "");
const ParameterString FeedForwardLayer::paramWeightDecay ("weight-decay", "");
const ParameterDouble FeedForwardLayer::paramWeightDecayFactor ("weight-decay-factor", 0.0f);

FeedForwardLayer::FeedForwardLayer(Configuration const& config) : NetworkLayer(config),
                                                                  nonlinearity_(nonlinearity_from_string(paramNonlinearity(config))),
                                                                  weight_decay_(weight_decay_from_string(paramWeightDecay(config))),
                                                                  weight_decay_factor_(paramWeightDecayFactor(config)) {
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

#pragma omp parallel for
  for (size_t time_idx = 0; time_idx < max_time_step; time_idx++) {
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
    std::cerr << "Output dimensions: " << slice.size()[1] << " x " << slice.size()[2] << std::endl;
    std::cerr << "Bias dimensions  : " << batch_size_ << " x " << output_size_ << std::endl;

    std::cerr << "Batch   size B: " << batch_size_ << std::endl;
    std::cerr << "Feature size D: " << feature_size_ << std::endl;
    std::cerr << "Output  size H: " << output_size_  << std::endl;
    std::cerr << "Size of bias   : " << bias.size() << std::endl;
    std::cerr << "Size of weights: " << weights.size() << std::endl;
    std::cerr << "Size of input  : " << input.size() << std::endl;

    std::cerr << "Bias (Tranposed) : " << std::endl;
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

    /*
    std::cerr << "Output: " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
        std::cerr << result[output_idx + batch_idx * output_size_] << " " ;
      }
      std::cerr << std::endl;
    }
		*/
    //applying different activation functions
    switch(nonlinearity_){
      case Nonlinearity::Sigmoid: {
        result = 1/(1+exp(-result));
        break;
      }
      case Nonlinearity::Tanh: {
      	result = 2/(1+exp(-2.0f*(result)))-1;
        break;
      }
      case Nonlinearity::ReLU:{
      	result = result.apply([](float n)->float { return std::max(n, 0.0f); });
        break;
      }
      case Nonlinearity::None: {
        break;
      }
    }

    output[output_slice] = result;
    /*
    std::cerr << "Output (after activations): " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
        std::cerr << result[output_idx + batch_idx * output_size_] << " " ;
      }
      std::cerr << std::endl;
    }
    */
	}
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
  std::valarray<float> ones (1.0f, slice.size()[1]* slice.size()[2]);
  std::valarray<float> inner_gradients(0.0f, slice.size()[1]* slice.size()[2]);

	std::valarray<float> weight_decay = params_[weights_slice];


  // Go over every time step
#pragma omp parallel for
  for (size_t time_idx = 0; time_idx < max_time_step; time_idx++) {
  	std::gslice input_slice(time_idx  * batch_size_ * feature_size_,
  			{batch_size_, feature_size_},
  			{feature_size_, 1});

  	std::valarray<float> input = input_buffer_[input_slice];

  	// Slice of size B x H
  	std::gslice output_slice(time_idx * slice.size()[1] * slice.size()[2],
  			{slice.size()[1], slice.size()[2]},
  			{slice.stride()[1], slice.stride()[2]});

  	std::valarray<float> current_error = error[output_slice];
  	std::valarray<float> out_slice     = output[output_slice];

  	std::valarray<float> inner_gradients(0.0f, slice.size()[1]* slice.size()[2]);
  	switch (nonlinearity_) {
  	case Nonlinearity::None:
  		inner_gradients = 1.0f;
  		break;
  	case Nonlinearity::Sigmoid:
  		inner_gradients = out_slice*(ones - out_slice);
  		break;
  	case Nonlinearity::Tanh:
  		inner_gradients = 1 - (out_slice * out_slice);
  		break;
  	case Nonlinearity::ReLU:
  		inner_gradients = out_slice.apply([](float n)->float {
  			if (n>0)
  				return 1;
  			else
  				return 0;
  		});
  		break;
  	default:
  		break;
  	}
  	current_error*=inner_gradients;

  	std::valarray<float> bias_gradient(0.0f, output_size_);
  	for (size_t i = 0;i<output_size_;i++){
  		std::gslice temp_slice(i,{batch_size_},{output_size_});
  		std::valarray<float> temp_error = current_error[temp_slice];
  		bias_gradient[i] = temp_error.sum();
  	}

  	std::gslice bias_slice(output_size_*feature_size_, {output_size_}, {1});
  	std::valarray<float> weight_gradients = gradient_[weights_slice];

  	/*
    std::cerr << "B: Error (transposed) : " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
        std::cerr << current_error[batch_idx * output_size_ + output_idx] << " " ;
      }
      std::cerr << std::endl;
    }

    std::cerr << "B: Input: " << std::endl;
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      for (size_t feature_idx = 0; feature_idx < feature_size_; feature_idx++) {
        std::cerr << input[feature_idx + batch_idx * feature_size_] << " " ;
      }
      std::cerr << std::endl;
    }
    */
  	// Perform the following calculation:
  	// gradient^(HxD) = current_error^(BxH) * input^(BxD)
  	// current error has to be transposed
  	cblas_sgemm(CblasRowMajor,
  			CblasTrans,
  			CblasNoTrans,
  			output_size_,
  			feature_size_,
  			batch_size_,
  			1.0f,
  			&current_error[0], // H x B
  			output_size_,      // # columns in the error (untransposed)
  			&input[0],         // B x D
  			feature_size_,     // # columns in the input
  			0.0f,
  			&weight_gradients[0],     // HxD
  			feature_size_);    // # columns in the gradient
  	/*
    std::cerr << "B: Output: " << std::endl;
    for (size_t output_idx = 0; output_idx < output_size_; output_idx++) {
      for (size_t feature_idx = 0; feature_idx < feature_size_; feature_idx++) {
        std::cerr << weight_gradients[feature_idx + output_idx * feature_size_] << " " ;
      }
      std::cerr << std::endl;
    }
		*/
  	if (input_error_needed_) {
  		// Perform the following calculation:
  		// error_buffer^(BxD) = current_error^(BxH) * weights^(HxD)
  		cblas_sgemm(CblasRowMajor,
  				CblasNoTrans,
  				CblasNoTrans,
  				batch_size_,
  				feature_size_,
  				output_size_,
  				1.0f,
  				&current_error[0], // B x H
  				output_size_,      // # columns in the current error
  				&weights[0],       // H x D
  				feature_size_,     // # columns in the weights
  				0.0f,
  				&error_buffer_[time_idx * feature_size_ * batch_size_],	// B x D new errors
  				feature_size_);    // # columns in the bias
  	}

  	switch (weight_decay_) {
  	case WeightDecay::L1: {
  		weight_decay  = params_[weights_slice];
  		weight_decay *= weight_decay_factor_ / 2;
  		weight_decay.apply([](float n)->float { return fabs(n); });
  		break;
  	}
  	case WeightDecay::L2: {
  		weight_decay  = params_[weights_slice];
  		weight_decay *= weight_decay_factor_;
  		break;
  	}
  	default:
  		weight_decay  = 0.0f;
  		break;
  	}
#pragma omp critical
  	{
  		gradient_[weights_slice] += weight_gradients + weight_decay;
  		gradient_[bias_slice]    += bias_gradient;
  	}
  }
}

FeedForwardLayer::FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity,
																	 WeightDecay weight_decay,
																	 float weight_decay_factor)
                                  : NetworkLayer(config), nonlinearity_(nonlinearity), weight_decay_(weight_decay),
                                    weight_decay_factor_(weight_decay_factor){
}

