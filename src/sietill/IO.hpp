/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __IO_H__
#define __IO_H__

#include <string>
#include <iostream>
#include <vector>

std::vector<short> read_audio_file(std::string const& path);
std::vector<float> read_feature_file(std::string const& path);

void write_to_txt_file(std::ostream& out, std::vector<double> const& vec, size_t start, size_t window_size, size_t sample_rate);
void write_floats_to_file(std::ostream& out, std::vector<double> const& data);

template<typename T>
void write_binary_blob(std::ostream& out, std::vector<T> const& data) {
  out.write(reinterpret_cast<char const*>(&data[0]), sizeof(T) * data.size());
}

template<typename T>
void read_binary_blob(std::istream& out, std::vector<T>& data) {
  out.read(reinterpret_cast<char*>(&data[0]), sizeof(T) * data.size());
}

#endif /* __IO_H__ */
