#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

namespace {
  double exp_add(double x, double y) {
    return x + exp(y);
  }

  template<typename T>
  struct exp_divide {
    T scale_;

    exp_divide(T const& scale) : scale_(scale) {}

    T operator()(T const& x) {
      return exp(x) / scale_;
    }
  };

}
void OutputLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  FeedForwardLayer::forward(output, slice, mask);

  unsigned max_time_step = *std::max_element(mask.begin(), mask.end());

  // Go over every time step
  for (size_t time_idx = 0; time_idx < max_seq_length_; time_idx++) {
    if (time_idx > max_time_step) {
      break;
    }

    // Go through every output vector in a batch
    for (size_t batch_idx = 0; batch_idx < batch_size_; batch_idx++) {
      // Slice of size H
      std::gslice output_slice(time_idx * slice.size()[1] * slice.size()[2] + batch_idx * slice.size()[2],
                             {slice.size()[2]},
                             {slice.stride()[2]});
      std::valarray<float> output_values(output[output_slice]);

      // TODO: This is the naïve computation of the soft-max function, which is supposed to be numerically unstable.
      // However, the subsequent sum over all elements is 1, showing no signs of instability?
      double sum = std::accumulate(std::begin(output_values), std::end(output_values), 0.0, exp_add);

      std::transform(std::begin(output_values), std::end(output_values),
                     std::begin(output_values), exp_divide<float>(sum));

      output[output_slice] = output_values;
      //std::cout << "Sum of elements: " << output_values.sum() << std::endl;
    }
  }
}

