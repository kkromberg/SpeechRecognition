/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __TYPES_HPP__
#define __TYPES_HPP__

#include <stdint.h>
#include <vector>
#include <utility>

typedef size_t   SegmentIdx;
typedef size_t   WordIdx;
typedef uint16_t StateIdx;
typedef uint16_t DensityIdx;

struct MixtureDensity {
  DensityIdx mean_idx;
  DensityIdx var_idx;

  MixtureDensity(DensityIdx mean_idx, DensityIdx var_idx)
                : mean_idx(mean_idx), var_idx(var_idx) {}
};

typedef std::vector<MixtureDensity> Mixture;

struct AlignmentItem {
  uint16_t count;
  StateIdx state;
  float    weight;

  AlignmentItem() : count(0u), state(0u), weight(0.0) {}

  AlignmentItem(uint16_t count, StateIdx state, float weight)
               : count(count), state(state), weight(weight) {}
};

typedef std::vector<AlignmentItem> Alignment;

typedef std::vector<WordIdx>::const_iterator WordIter;

#endif /* TYPES_HPP */
