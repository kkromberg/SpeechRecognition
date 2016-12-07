/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __NETWORK_LAYER_HPP__
#define __NETWORK_LAYER_HPP__

#include <valarray>

#include "Config.hpp"

class NetworkLayer {
public:
  static const ParameterString paramLayerName;
  static const ParameterUInt   paramOutputSize;

  NetworkLayer(Configuration const& config);
  virtual ~NetworkLayer();

  virtual void init(bool input_error_needed);

  std::string const& get_layer_name() const;
  void               set_input_sizes(size_t feature_size, size_t batch_size, size_t max_seq_length);
  size_t             get_output_size() const;

  std::vector<std::string> const& get_input_layer_names() const;

  std::valarray<float>& get_input_buffer();
  std::valarray<float>& get_error_buffer();
  std::valarray<float>& get_params();
  std::valarray<float>& get_gradient();

  virtual void init_parameters(std::function<float()> const& generator) = 0;
  virtual void forward (std::valarray<float>& output,
                        std::gslice const& slice, std::vector<unsigned> const& mask) const = 0;
  virtual void backward_start() = 0;
  virtual void backward(std::valarray<float>& output, std::valarray<float>& error,
                        std::gslice const& slice, std::vector<unsigned> const& mask) = 0;
  virtual void save(std::string const& path) const;
  virtual void load(std::string const& path);

  const size_t get_feature_size()    const { return feature_size_; };
  const size_t get_max_seq_length_() const { return max_seq_length_; };
protected:
  const std::string layer_name_;

  size_t       feature_size_;
  size_t       batch_size_;
  size_t       max_seq_length_;
  const size_t output_size_;
  bool         input_error_needed_;

  std::vector<std::string> input_layer_names_;

  std::valarray<float> input_buffer_;
  std::valarray<float> error_buffer_;

  std::valarray<float> params_;
  std::valarray<float> gradient_;
};

// Some methods are defined in the header to allow inlining

inline NetworkLayer::NetworkLayer(Configuration const& config)
                                 : layer_name_(paramLayerName(config)),
                                   feature_size_(0ul), // these 3 are set later via set_input_sizes
                                   batch_size_(0ul),
                                   max_seq_length_(0ul),
                                   output_size_(paramOutputSize(config)),
                                   input_layer_names_(config.get_string_array("input")) {
}
 
inline NetworkLayer::~NetworkLayer() {
}

inline std::string const& NetworkLayer::get_layer_name() const {
  return layer_name_;
}

inline void NetworkLayer::set_input_sizes(size_t feature_size, size_t batch_size, size_t max_seq_length) {
  feature_size_   = feature_size;
  batch_size_     = batch_size;
  max_seq_length_ = max_seq_length;
  input_buffer_.resize(feature_size * batch_size * max_seq_length);
  error_buffer_.resize(feature_size * batch_size * max_seq_length);
}

inline size_t NetworkLayer::get_output_size() const {
  return output_size_;
}

inline std::vector<std::string> const& NetworkLayer::get_input_layer_names() const {
  return input_layer_names_;
}

inline std::valarray<float>& NetworkLayer::get_input_buffer() {
  return input_buffer_;
}

inline std::valarray<float>& NetworkLayer::get_error_buffer() {
  return error_buffer_;
}

inline std::valarray<float>& NetworkLayer::get_params() {
  return params_;
}

inline std::valarray<float>& NetworkLayer::get_gradient() {
  return gradient_;
}

#endif /* __NETWORK_LAYER_HPP__ */
