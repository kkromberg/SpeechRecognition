#include "OutputLayer.hpp"

#include <iostream>

#include "Util.hpp"

void OutputLayer::forward(std::valarray<float>& output, std::gslice const& slice, std::vector<unsigned> const& mask) const {
  FeedForwardLayer::forward(output, slice, mask);

  // TODO: Implement soft-max
}

