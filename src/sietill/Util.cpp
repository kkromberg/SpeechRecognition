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
#include <math.h>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <map>

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

void write_logspcetrum_to_file(std::string const& filename, std::vector<double> const& vec){
  std::ofstream file;
  file.open(filename.c_str(), std::ios_base::app);
  for (auto i = vec.begin(); i != vec.end(); ++i) {
    file << log10(*i) <<",";
  }
  file << "\n";
  file.close();
}

void calculate_energy(std::string const& filename, std::vector<double> const& vec){
  std::ofstream file;
  file.open(filename.c_str(), std::ios_base::app);
  double energy = 0;
  for (auto i = vec.begin(); i != vec.end(); ++i) {
    energy += log10(*i);
  }
  file << energy << "\n";
  file.close();

}
void create_pgm(std::string const& input_file, std::string const& output_file){
  // load txt file
  std::ifstream txt_file;
  txt_file.open(input_file.c_str());
  std::string line;
  std::map<int, std::vector<double>> all_values;
  std::vector<double> log_spectrum;
  // read all
  double max = 0;
  double min = 0;
  int vector_counter = 0;
  int value_counter = 0;
  while (std::getline(txt_file, line)) {
    std::istringstream ss(line);
    std::string value;
    // store each value in vector
    value_counter = 0;
    while (std::getline(ss, value, ',')) {
      log_spectrum.insert(log_spectrum.begin()+value_counter, (atof(value.c_str())));
      value_counter++;
    }
    // get max
    if (*max_element(log_spectrum.begin(), log_spectrum.end()) > max) {
      max = *max_element(log_spectrum.begin(), log_spectrum.end());
    }
    // get min
    if (*min_element(log_spectrum.begin(), log_spectrum.end()) < min) {
      min = *min_element(log_spectrum.begin(), log_spectrum.end());
    }
    // put vector into map
    all_values[vector_counter] = log_spectrum;
    vector_counter++;

  }
  //std::cout << "Current max: " << max << "\n Current min: " << min << "\n";

  // create pgm header
  std::string pgm_header = "P2\n" + std::to_string(vector_counter) + " " + std::to_string(value_counter-1) + " 255\n";
  //std::cout << pgm_header;
  std::ofstream pgm_file;
  pgm_file.open(output_file.c_str());
  pgm_file << pgm_header;

  std::ofstream tmp_file;
  tmp_file.open("tmp_file.txt");
  tmp_file << pgm_header;

  // iterate through all the vectors and write them to file
  for (int row = 0; row < value_counter -1; row++) {
    std::string line = "";
    for (int column = 0; column < vector_counter; column++) {
      int scaled_value = (int)((255) * (all_values[column][row] - min))/(max-min);
        line += std::to_string(scaled_value) + " ";
    }
      pgm_file << line + "\n";
  }
  pgm_file.close();

}
