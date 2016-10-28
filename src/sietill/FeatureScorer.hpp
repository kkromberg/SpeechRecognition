/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __FEATURE_SCORER_HPP__
#define __FEATURE_SCORER_HPP__

#include "Iter.hpp"

class FeatureScorer {
public:
  virtual void prepare_sequence(FeatureIter const& start, FeatureIter const& end) = 0;
  virtual double score(FeatureIter const& iter, StateIdx state_idx) const = 0;
};

#endif /* __FEATURE_SCORER_HPP__ */
