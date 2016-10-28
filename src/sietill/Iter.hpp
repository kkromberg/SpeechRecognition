/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __ITER_HPP__ 
#define __ITER_HPP__ 

#include <assert.h>
#include <iterator>

#include "Types.hpp"

template<typename T>
class Iter2D : public std::iterator<std::random_access_iterator_tag, T*> {
public:
  typedef typename std::iterator<std::random_access_iterator_tag, T*>::difference_type difference_type;

  const size_t size;

  Iter2D(T* pos, size_t size) : size(size), pos_(pos) {}
  Iter2D(Iter2D<T> const& other) : size(other.size), pos_(other.pos_) {}

  // iterator concept
  
  Iter2D<T>& operator=(Iter2D<T> const& other) {
    assert(size == other.size);
    pos_ = other.pos_;
    return *this;
  }

  T* operator*() const {
    return pos_;
  }

  Iter2D& operator++() {
    pos_ += size;
    return *this;
  }

  // input/forward iterator concept
  bool operator!=(Iter2D const& it) const {
    return (pos_ != it.pos_) or (size != it.size);
  }

  Iter2D operator++(int _) {
    Iter2D res(pos_, size);
    pos_ += size;
    return res;
  }

  // bidirectional iterator concept
  Iter2D& operator--() {
    pos_ -= size;
    return *this;
  }

  Iter2D operator--(int _) {
    Iter2D res(pos_, size);
    pos_ -= size;
    return res;
  }

  // random access iterator concept
  Iter2D& operator+=(difference_type const& diff) {
    pos_ += diff * size;
    return *this;
  }

  Iter2D operator+(difference_type const& diff) const {
    return Iter2D(pos_ + diff * size, size);
  }

  Iter2D& operator-=(difference_type const& diff) {
    pos_ -= diff * size;
    return *this;
  }

  Iter2D operator-(difference_type const& diff) const {
    return Iter2D(pos_ - diff * size, size);
  }

  difference_type operator-(Iter2D const& other) const {
    assert(size == other.size);
    return (pos_ - other.pos_) / size;
  }

  T* operator[](difference_type const& idx) const {
    return pos_ + idx * size;
  }

  bool operator< (Iter2D const& it) const { return pos_ <  it.pos_; };
  bool operator> (Iter2D const& it) const { return pos_ >  it.pos_; };
  bool operator>=(Iter2D const& it) const { return pos_ <= it.pos_; };
  bool operator<=(Iter2D const& it) const { return pos_ >= it.pos_; };

private:
  T* pos_;
};

typedef Iter2D<const float> FeatureIter;
typedef Iter2D<AlignmentItem> AlignmentIter;
typedef Iter2D<const AlignmentItem> ConstAlignmentIter;

#endif /* __ITER_HPP__ */
