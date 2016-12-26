/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "NeuralNetwork.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include "FeedForwardLayer.hpp"
#include "Util.hpp"

namespace {
  static const ParameterString paramLayerType("type", "");

  std::unique_ptr<NetworkLayer> layer_from_config(Configuration const& config) {
    std::string type = paramLayerType(config);
    if (type == "feed-forward") {
      return std::unique_ptr<NetworkLayer>(new FeedForwardLayer(config));
    }
    else if (type == "output") {
      return std::unique_ptr<NetworkLayer>(new OutputLayer(config));
    }
    std::cerr << "unknown layer-type: " << type << std::endl;
    return std::unique_ptr<NetworkLayer>();
  }

  void build_single_sequence(FeatureIter begin, FeatureIter const& end, size_t base_feature_size,
                             size_t context_frames, std::valarray<float>& output, std::gslice const& slice) {
    int num_frames = static_cast<int>(std::distance(begin, end));
    size_t center_time = 0ul;
    while (begin != end) {
      for (int delta = -context_frames; delta <= static_cast<int>(context_frames); delta++) {
        const int feature_time = static_cast<int>(center_time) + delta;
        if (feature_time < 0 or feature_time >= num_frames) {
          continue;
        }
        for (size_t f = 0ul; f < begin.size; f++) {
          size_t index =   slice.start()
                         + (center_time + delta)                              * slice.stride()[0]
                         + ((delta + context_frames) * base_feature_size + f) * slice.stride()[2];
          float val = (*begin)[f];
          output[index] = val;
        }
      }
      ++begin;
      ++center_time;
    }
  }
}

const ParameterString NeuralNetwork::paramLoadNeuralNetworkFrom("load-nn-from", "");
const ParameterString NeuralNetwork::paramPriorFile            ("prior-file", "");
const ParameterFloat  NeuralNetwork::paramPriorScale           ("prior-scale", 0.0f);
const ParameterUInt   NeuralNetwork::paramContextFrames        ("context-frames", 0); // this duplicates a parameter from MinibatchBuilder, it is only used if the NN is used as a feature-scorer

NeuralNetwork::NeuralNetwork(Configuration const& config, size_t feature_size, size_t batch_size,
                             size_t max_seq_length, size_t num_classes)
                            : feature_size_(feature_size), batch_size_(batch_size), max_seq_length_(max_seq_length),
                              num_classes_(num_classes), context_frames_(paramContextFrames(config)),
                              initialized_(false), feature_buffer_(0.0f, feature_size * batch_size * max_seq_length),
                              feature_buffer_start_(&feature_buffer_[0], feature_size / (2ul * context_frames_ + 1ul)),
                              batch_mask_(batch_size, 0ul),
                              score_buffer_(max_seq_length * batch_size * num_classes),
                              error_buffer_(max_seq_length * batch_size * num_classes),
                              prior_path_(paramPriorFile(config)), prior_scale_(paramPriorScale(config)),
                              log_prior_(num_classes) {
  // first we create the layers from configs
  std::vector<Configuration> layer_configs = config.get_array("layers");
  layers_.resize(layer_configs.size());
  output_infos_.resize(layer_configs.size() + 1u);
  for (size_t i = 0ul; i < layer_configs.size(); i++) {
    layers_[i] = layer_from_config(layer_configs[i]);
  }

  // now we need to sort the layer s.t. we can evaluate them in the correct order
  std::vector<unsigned> topo_index(layers_.size(), 0u);
  std::map<std::string, unsigned> name_to_idx;

  for (size_t i = 0ul; i < layers_.size(); i++) {
    if (name_to_idx.find(layers_[i]->get_layer_name()) != name_to_idx.end()) {
      std::cerr << "Duplicate layer name: " << layers_[i]->get_layer_name() << std::endl;
      std::abort();
    }
    name_to_idx[layers_[i]->get_layer_name()] = i;
  }

  // we do very simple and slow O(n^2) topo-sort
  // as the number of layers is not that large it should not matter much
  for (size_t i = 0ul; i < layers_.size(); i++) {
    for (size_t l = 0ul; l < layers_.size(); l++) {
      for (std::string const& input : layers_[l]->get_input_layer_names()) {
        if (input == "data") {
          topo_index[l] = std::max(topo_index[l], 1u);
        }
        else {
          auto iter = name_to_idx.find(input);
          if (iter == name_to_idx.end()) {
            std::cerr << "Unknown input: " << input << std::endl;
            std::abort();
          }
          topo_index[l] = std::max(topo_index[l], topo_index[iter->second]);
        }
      }
    }
  }

  std::sort(layers_.begin(), layers_.end(), [&](std::unique_ptr<NetworkLayer> const& a, std::unique_ptr<NetworkLayer> const& b){
    unsigned topo_idx_a = topo_index[name_to_idx[a->get_layer_name()]];
    unsigned topo_idx_b = topo_index[name_to_idx[b->get_layer_name()]];
    return topo_idx_a < topo_idx_b;
  });

  // rebuild name_to_idx
  for (size_t i = 0ul; i < layers_.size(); i++) {
    name_to_idx[layers_[i]->get_layer_name()] = i;
  }

  std::vector<bool> input_error_needed(layers_.size(), false);

  // now we can build the output infos
  for (size_t l = 0ul; l < layers_.size(); l++) {
    size_t fdim = 0ul;
    for (std::string const& input : layers_[l]->get_input_layer_names()) {
      if (input == "data") {
        fdim += feature_size_;
      }
      else {
        size_t idx = name_to_idx[input];
        fdim += layers_[idx]->get_output_size();
        input_error_needed[l] = true;
      }
    }

    layers_[l]->set_input_sizes(fdim, batch_size_, max_seq_length_);
    std::valarray<float>& input_buffer = layers_[l]->get_input_buffer();
    std::valarray<float>& error_buffer = layers_[l]->get_error_buffer();
    size_t offset = 0ul;
    for (std::string const& input : layers_[l]->get_input_layer_names()) {
      if (input == "data") {
        output_infos_[0].push_back(OutputBuffer(input_buffer,
                                                error_buffer,
                                                std::gslice(offset,
                                                            {   max_seq_length_, batch_size_, feature_size_},
                                                            {batch_size_ * fdim,        fdim,             1})));
        offset += feature_size_;
      }
      else {
        size_t idx = name_to_idx[input];
        output_infos_[idx + 1].push_back(OutputBuffer(input_buffer,
                                                      error_buffer,
                                                      std::gslice(offset,
                                                                  {   max_seq_length_, batch_size_, layers_[idx]->get_output_size()},
                                                                  {batch_size_ * fdim,        fdim,                               1})));
        offset += layers_[idx]->get_output_size();
      }
    }
    layers_[l]->init(input_error_needed[l]);
    parameters_[layers_[l]->get_layer_name()] = &layers_[l]->get_params();
    gradients_ [layers_[l]->get_layer_name()] = &layers_[l]->get_gradient();
  }
  output_infos_.back().push_back(OutputBuffer(score_buffer_,
                                              error_buffer_,
                                              std::gslice(0,
                                                          { max_seq_length_, batch_size_, num_classes_ },
                                                          { batch_size_ * num_classes_, num_classes_, 1ul })));

  std::string path = paramLoadNeuralNetworkFrom(config);
  if (not path.empty()) {
    load(path);
  }

  log_prior_ = 0.0f;
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::prepare_sequence(FeatureIter const& start, FeatureIter const& end) {
  feature_buffer_start_ = start;
  build_single_sequence(start, end, feature_size_ / (2ul * context_frames_ + 1ul),
                        context_frames_, feature_buffer_, get_feature_buffer_slice());
  batch_mask_[0ul] = std::distance(start, end);
  forward();
  score_buffer_ = -std::log(score_buffer_);
  for (size_t t = 0ul; t < batch_mask_[0ul]; t++) {
    score_buffer_[std::slice(t * num_classes_, num_classes_, 1ul)] += log_prior_;
  }
}

double NeuralNetwork::score(FeatureIter const& iter, StateIdx state_idx) const {
  size_t frame = std::distance(feature_buffer_start_, iter);
  return score_buffer_[frame * num_classes_ + state_idx];
}

std::valarray<float>& NeuralNetwork::get_feature_buffer() {
  return feature_buffer_;
}

std::gslice NeuralNetwork::get_feature_buffer_slice() const {
  return std::gslice(0ul,
                    {max_seq_length_, batch_size_, feature_size_},
                    {batch_size_ * feature_size_, feature_size_, 1ul});
}

std::vector<unsigned>& NeuralNetwork::get_batch_mask() {
  return batch_mask_;
}

std::valarray<float> const& NeuralNetwork::get_score_buffer() const {
  return score_buffer_;
}

NetworkLayer* NeuralNetwork::get_network_layer(std::string const& name) {
  for (auto& layer : layers_) {
    if (layer->get_layer_name() == name) {
      return layer.get();
    }
  }
  return nullptr;
}

std::map<std::string, std::valarray<float>*> const& NeuralNetwork::get_parameters() const {
  return parameters_;
}

std::map<std::string, std::valarray<float>*> const& NeuralNetwork::get_gradients() const {
  return gradients_;
}

void NeuralNetwork::init_parameters(std::function<float()> const& generator) {
  if (not initialized_) {
    for (auto& layer : layers_) {
      layer->init_parameters(generator);
    }
    initialized_ = true;
  }
}

void NeuralNetwork::forward() {
  // set input features
  for (auto& out : output_infos_[0ul]) {
    out.fwd_buffer[out.slice] = feature_buffer_;
  }
  for (size_t l = 0ul; l < layers_.size(); l++) {
    OutputBuffer& out = output_infos_[l + 1ul][0ul];
    layers_[l]->forward(out.fwd_buffer, out.slice, batch_mask_);
    for (size_t o = 1ul; o < output_infos_[l + 1ul].size(); o++) {
      OutputBuffer& out2 = output_infos_[l + 1ul][o];
      out.fwd_buffer[out.slice] = out2.fwd_buffer[out2.slice];
    }
  }
}

void NeuralNetwork::backward(std::valarray<float> const& targets) {
  if (score_buffer_.size() != targets.size()) {
    std::cerr << "target size is wrong. is: " << targets.size() << " should be: " << score_buffer_.size() << std::endl;
    abort();
  }

  error_buffer_ = score_buffer_ - targets; // we compute the gradient for the softmax output here
  for (size_t l = layers_.size(); l > 0ul; l--) {
    layers_[l-1ul]->backward_start();
    for (size_t o = 0ul; o < output_infos_[l].size(); o++) {
      OutputBuffer& out = output_infos_[l][o];
      layers_[l-1ul]->backward(out.fwd_buffer, out.bwd_buffer, out.slice, batch_mask_);
    }
  }
}

void NeuralNetwork::save(std::string const& folder) const {
  for (size_t l = 0ul; l < layers_.size(); l++) {
    layers_[l]->save(folder + layers_[l]->get_layer_name());
  }
}

void NeuralNetwork::load(std::string const& folder) {
  for (size_t l = 0ul; l < layers_.size(); l++) {
    layers_[l]->load(folder + layers_[l]->get_layer_name());
  }
  initialized_ = true;
}

void NeuralNetwork::load_prior() {
  load_prior(prior_path_);
}

void NeuralNetwork::load_prior(std::string const& path) {
  std::ifstream input(path, std::ios::in);
  if (not input.good()) {
    std::cerr << "Could not load prior: " << path << std::endl;
    std::abort();
  }

  size_t idx = 0ul;
  while (input.good() and not input.eof() and idx < log_prior_.size()) {
    input >> log_prior_[idx++];
  }
  log_prior_ = prior_scale_ * std::log(log_prior_);
}

void NeuralNetwork::gradient_test() {
  for (size_t l = 0ul; l < layers_.size(); l++) {
    layers_[l]->gradient_test();
  }
}


