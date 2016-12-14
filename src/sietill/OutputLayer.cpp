#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

namespace {
  template<typename T>
  struct shifted_exp_add {
  	T shift_;

  	shifted_exp_add(T const& shift) : shift_(shift) {}

  	T operator()(T const& prev, T const& x) {
  		return prev + exp(x - shift_);
  	}
  };

  template<typename T>
  struct exp_minus {
  	T value_;

  	exp_minus(T const& value) : value_(value) {}

  	T operator()(T const& x) {
  		return exp(x - value_);
  	}
  };
}
void OutputLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  FeedForwardLayer::forward(output, slice, mask);

  unsigned max_time_step = *std::max_element(mask.begin(), mask.end());

  // Go over every time step
#pragma omp parallel for
  for (size_t time_idx = 0; time_idx < max_time_step; time_idx++) {

    // Go through every output vector in a batch
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {

      // Slice of size H
      std::gslice output_slice(time_idx * slice.size()[1] * slice.size()[2] + batch_idx * slice.size()[2],
                             {slice.size()[2]},
                             {slice.stride()[2]});
      std::valarray<float> output_values(output[output_slice]);

    	if (time_idx >= mask[batch_idx]) {
    		output[output_slice] = 0.0f;
    	} else {
        // calculate the output layer in log-space to avoid numerical instability
        // see: https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
        double max_value = *std::max_element(std::begin(output_values), std::end(output_values));
        double log_space_sum = std::accumulate(std::begin(output_values), std::end(output_values),
        																			 0.0, shifted_exp_add<double>(max_value));
        log_space_sum = max_value + log(log_space_sum);
        std::transform(std::begin(output_values), std::end(output_values),
                       std::begin(output_values), exp_minus<float>(log_space_sum));

        output[output_slice] = output_values;
    	}

      //std::cout << "Sum of elements: " << output_values.sum() << std::endl;
    }
  }
}

