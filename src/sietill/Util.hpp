/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <string>
#include <vector>

void create_dir(std::string const& str);

void dump_matrix    (std::string const& filename, float const* data, size_t n, size_t m, size_t lda);
void dump_int_matrix(std::string const& filename, unsigned const* data, size_t n, size_t m, size_t lda);

void write_logspcetrum_to_file(std::string const& filename, std::vector<double> const& vec);
void create_pgm(std::string const& input_file, std::string const& output_file);
void calculate_energy(std::string const& filename, std::vector<double> const& vec, int row);

#endif /* __UTIL_HPP__ */
