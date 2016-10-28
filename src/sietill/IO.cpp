/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <fstream>

#include "IO.hpp"

/*****************************************************************************/

std::vector<short> read_audio_file(std::string const& path) {
  std::vector<short> samples;

  /* open sph/wav file */
  std::ifstream input_stream(path.c_str(), std::ios_base::in | std::ios_base::binary);
  if (not input_stream.good()) {
    std::cerr << "Error: cannot open " << path << std::endl;
    return samples;
  }

  char header[5] = { 0, 0, 0, 0, 0 };
  input_stream.read(header, sizeof(header) - 1);
  if (std::string(header) == "RIFF") {
    input_stream.seekg(44, std::ios_base::beg);
  }
  else {
    /* skip 1024 byte header for .sph files */
    input_stream.seekg(1024, std::ios_base::beg);
  }

  short buffer[2048];
  while (input_stream.good()) {
    input_stream.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    size_t samples_read = input_stream.gcount() / sizeof(short);
    size_t insert_pos   = samples.size();

    samples.resize(samples.size() + samples_read);
    std::copy(buffer, buffer + samples_read, &samples[insert_pos]);
  }

  return samples;
}

/*****************************************************************************/

std::vector<float> read_feature_file(std::string const& path) {
  std::vector<float> features;

  std::ifstream input_stream(path.c_str(), std::ios_base::in | std::ios_base::binary);
  if (not input_stream.good()) {
    std::cerr << "Error: cannot open " << path << std::endl;
    return features;
  }

  float buffer[512];
  while (input_stream.good()) {
    input_stream.read(reinterpret_cast<char*>(buffer), sizeof(buffer));
    size_t features_read = input_stream.gcount() / sizeof(float);
    size_t insert_pos    = features.size();

    features.resize(features.size() + features_read);
    std::copy(buffer, buffer + features_read, &features[insert_pos]);
  }

  return features;
}

/*****************************************************************************/

void write_to_txt_file(std::ostream& out, std::vector<double> const& vec, size_t start, size_t window_size, size_t sample_rate) {
  out << static_cast<double>(start + window_size / 2) / static_cast<double>(sample_rate);
  for (size_t idx = 0u; idx < vec.size(); idx++) {
    out << " " << vec[idx];
  }
  out << std::endl;
}

/*****************************************************************************/

void write_floats_to_file(std::ostream& out, std::vector<double> const& data) {
  static std::vector<float> buffer;

  if (buffer.size() != data.size()) {
    buffer.resize(data.size());
  }

  std::copy(data.begin(), data.end(), buffer.begin());

  out.write(reinterpret_cast<char const*>(&buffer[0]), sizeof(float) * buffer.size());
}

/*****************************************************************************/

