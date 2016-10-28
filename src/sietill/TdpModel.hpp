/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __TDP_MODEL_HPP__
#define __TDP_MODEL_HPP__

#include "Config.hpp"
#include "Types.hpp"

class TdpModel {
public:
  static const ParameterDouble paramLoop;
  static const ParameterDouble paramForward;
  static const ParameterDouble paramSkip;

  const StateIdx silence_state;

  TdpModel(Configuration const& config, StateIdx silence_state);
  ~TdpModel();

  double score(StateIdx to, size_t jump) const;
private:
  const double tdp_loop_;
  const double tdp_forward_;
  const double tdp_skip_;
};

#endif /* __TDP_MODEL_HPP__ */
