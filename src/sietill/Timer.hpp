/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <time.h>

class Timer {
public:
  Timer() : diff_sec_(0u), diff_nsec_(0u) {}

  void tick() {
    clock_gettime(CLOCK_MONOTONIC, &start_);
  }

  void tock() {
    clock_gettime(CLOCK_MONOTONIC, &end_);
    diff_nsec_ += end_.tv_nsec - start_.tv_nsec;
    diff_sec_  += end_.tv_sec - start_.tv_sec + diff_nsec_ / 1000000000;
    diff_nsec_ %= 1000000000;
  }

  double secs() const {
    return diff_sec_ + static_cast<double>(diff_nsec_) / 1e9;
  }

  void reset() {
    diff_sec_  = 0ul;
    diff_nsec_ = 0ul;
  }
  
private:
  timespec start_;
  timespec end_;
  time_t   diff_sec_;
  long     diff_nsec_;
};

#endif /* __TIMER_HPP__ */
