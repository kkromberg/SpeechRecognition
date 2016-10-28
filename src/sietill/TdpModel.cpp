#include "TdpModel.hpp"

#include <limits>

const ParameterDouble TdpModel::paramLoop   ("tdp-loop",    0.0);
const ParameterDouble TdpModel::paramForward("tdp-forward", 0.0);
const ParameterDouble TdpModel::paramSkip   ("tdp-skip",    0.0);

TdpModel::TdpModel(Configuration const& config, StateIdx silence_state)
                  : silence_state(silence_state),
                    tdp_loop_   (paramLoop(config)),
                    tdp_forward_(paramForward(config)),
                    tdp_skip_   (paramSkip(config)) {
}

TdpModel::~TdpModel() {
}

double TdpModel::score(StateIdx to, size_t jump) const {
  if (to == silence_state) {
    return tdp_forward_;
  }
  switch (jump) {
    case 0: return tdp_loop_;
    case 1: return tdp_forward_;
    case 2: return tdp_skip_;
  }
  return std::numeric_limits<double>::infinity();
}
