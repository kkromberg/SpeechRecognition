/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "NetworkLayer.hpp"

#include <fstream>
#include <iostream>

#include "Util.hpp"

const ParameterString NetworkLayer::paramLayerName ("layer-name", "");
const ParameterUInt   NetworkLayer::paramOutputSize("num-outputs", 0u);

void NetworkLayer::init(bool input_error_needed) {
  input_error_needed_ = input_error_needed;
}

void NetworkLayer::save(std::string const& path) const {
  create_dir(path);
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&params_[0]), params_.size() * sizeof(float));
}

void NetworkLayer::load(std::string const& path) {
  std::ifstream in(path, std::ios::in);
  if (not in.good()) {
    std::cerr << "error loading parameters: " << path << std::endl;
    std::abort();
  }
  in.read(reinterpret_cast<char*>(&params_[0]), params_.size() * sizeof(float));
}
