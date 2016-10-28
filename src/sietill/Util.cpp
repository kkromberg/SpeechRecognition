/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Util.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>

#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>

void create_dir(std::string const& str) {
  std::vector<std::string> stack;

  // we need a modifiable c-string for dirname
  std::unique_ptr<char> path_buffer(new char[str.size()]);
  std::memcpy(path_buffer.get(), str.c_str(), str.size());
  char* d = dirname(path_buffer.get());

  while (mkdir(d, 0755) == -1) {
    if (errno != ENOENT) {
      if (errno != EEXIST) {
        std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
      }
      break;
    }
    stack.push_back(std::string(d));
    d = dirname(d);
  }

  while (not stack.empty()) {
    std::cerr << stack.back() << std::endl;
    mkdir(stack.back().c_str(), 0755);
    stack.pop_back();
  }
}

void dump_matrix(std::string const& filename, float const* data, size_t n, size_t m, size_t lda) {
  std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
  for (size_t i = 0ul; i < n; i++) {
    for (size_t j = 0ul; j < m; j++) {
      out << std::setprecision(10) << data[i * lda + j] << ' ';
    }
    out << '\n';
  }
}

void dump_int_matrix(std::string const& filename, unsigned const* data, size_t n, size_t m, size_t lda) {
  std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
  for (size_t i = 0ul; i < n; i++) {
    for (size_t j = 0ul; j < m; j++) {
      out << data[i * lda + j] << ' ';
    }
    out << '\n';
  }
}
