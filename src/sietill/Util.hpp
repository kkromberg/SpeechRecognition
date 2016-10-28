/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <string>

void create_dir(std::string const& str);

void dump_matrix    (std::string const& filename, float const* data, size_t n, size_t m, size_t lda);
void dump_int_matrix(std::string const& filename, unsigned const* data, size_t n, size_t m, size_t lda);

#endif /* __UTIL_HPP__ */
