#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

#include "xmmintrin.h"
#include "emmintrin.h"
#include "pmmintrin.h"

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

        //softmax_(output_values);
        output[output_slice] = output_values;
    	}

      //std::cout << "Sum of elements: " << output_values.sum() << std::endl;
    }
  }
}

float OutputLayer::shifted_exp_sum_(const float shift_value, std::valarray<float>& values) const {
	size_t loop_remainder = values.size() % 4;

	float buffer[4];
	__m128 shift = _mm_set_ps1(shift_value);
	__m128 vsum = _mm_set_ps1(0.0f);
	for (size_t i = 0; i < values.size() - loop_remainder; i += 4) {

		// perform the shift of the values in SSE registers
		__m128 outputs = _mm_loadu_ps(&values[i]);
		outputs = _mm_sub_ps(outputs, shift);

		// store the shifted values in an array and compute the exponential
		_mm_store_ps(&buffer[0], outputs);
		for (size_t j = 0; j < 4; j++) {
			buffer[j] = exp(buffer[j]);
		}

		// create an SSE register and add to the previous values
		outputs = _mm_loadu_ps(&buffer[0]);
		vsum = _mm_add_ps(vsum, outputs);
	}

	// add all values in the SSE register
	__m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
	__m128 sums = _mm_add_ps(vsum, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);
	float log_space_sum = _mm_cvtss_f32(sums);

	// Handle the loop remainder (without SSE registers) if necessary
	if (loop_remainder > 0) {
		for (size_t i = values.size() - loop_remainder; i < values.size(); i++) {
			log_space_sum += exp(values[i] - shift_value);
		}
	}

	return log_space_sum;
}

void OutputLayer::log_space_normalization_(const float norm, std::valarray<float>& values) const {
	size_t loop_remainder = values.size() % 4;

	__m128 shift = _mm_set_ps1(norm);
	for (size_t i = 0; i < values.size() - loop_remainder; i += 4) {
			// perform the shift of the values in SSE registers
			__m128 outputs = _mm_loadu_ps(&values[i]);
			outputs = _mm_sub_ps(outputs, shift);

			// store the shifted values in an array and compute the exponential
			_mm_store_ps(&values[i], outputs);
			for (size_t j = 0; j < 4; j++) {
				values[i+j] = exp(values[i+j]);
			}
		}

		// Handle the loop remainder (without SSE registers) if necessary
		if (loop_remainder > 0) {
			for (size_t i = values.size() - loop_remainder; i < values.size(); i++) {
				values[i] = exp(values[i] - norm);
			}
		}
}

void OutputLayer::softmax_(std::valarray<float>& values) const {
	float max_value     = *std::max_element(std::begin(values), std::end(values));
	float log_space_sum = max_value + log(shifted_exp_sum_(max_value, values));
	log_space_normalization_(log_space_sum, values);
}

