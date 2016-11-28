/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __NN_TRAINING_HPP__
#define __NN_TRAINING_HPP__

#include <random>
#include <valarray>

#include "Config.hpp"
#include "Corpus.hpp"
#include "NeuralNetwork.hpp"
#include "Mixtures.hpp"

class MiniBatchBuilder {
public:
  static const ParameterUInt   paramContextFrames;
  static const ParameterUInt   paramSeed;
  static const ParameterUInt   paramMaxSilenceFrames;
  static const ParameterBool   paramNormalizeFeaturesPerBatch;
  static const ParameterString paramTargetFile;
  static const ParameterFloat  paramCVSize;

  MiniBatchBuilder(Configuration const& config, Corpus const& corpus, size_t batch_size, size_t num_classes, StateIdx silence_state);
  virtual ~MiniBatchBuilder();

  size_t batch_size() const;
  size_t num_classes() const;
  size_t max_seq_length() const;
  size_t context_frames() const;
  size_t feature_size() const;
  size_t num_train_batches() const;
  size_t num_cv_batches() const;

  void shuffle();
  void build_batch(size_t batch_index, bool cv, std::valarray<float>& output, std::gslice const& slice,
                   std::valarray<float>& targets, std::vector<unsigned>& mask) const;
private:
  const size_t      batch_size_;
  const size_t      num_classes_;
  const size_t      max_seq_length_;
  const size_t      base_feature_size_;
  const size_t      context_frames_;
  const StateIdx    silence_state_;
  const size_t      max_silence_frames_;
  const bool        normalize_features_per_batch_;
  const std::string target_file_;
  const float       cv_size_;
  const size_t      num_train_seq_;
  const size_t      num_cv_seq_;
  const size_t      num_train_batches_;
  const size_t      num_cv_batches_;

  Corpus const& corpus_;
  Alignment     alignment_;
  size_t        max_aligns_;

  std::vector<size_t> train_segment_sequence_;
  std::vector<size_t> cv_segment_sequence_;
  std::mt19937        rng_;

  mutable std::valarray<float> mean_;
  mutable std::valarray<float> next_mean_;
  mutable std::valarray<float> sqr_;
  mutable std::valarray<float> diff_;
  mutable std::valarray<float> diff2_;

  std::pair<size_t, size_t> sequence_boundaries(size_t begin, size_t end) const;
};

class ParameterUpdater {
public:
  typedef std::map<std::string, std::valarray<float>*> VAMap;

  ParameterUpdater(VAMap const& parameters, VAMap const& gradients)
                  : parameters_(parameters), gradients_(gradients) {
  }
  virtual ~ParameterUpdater() {
  }

  virtual void update() = 0;
protected:
  VAMap const& parameters_;
  VAMap const& gradients_;
};

class SGDUpdater : public ParameterUpdater {
public:
  static const ParameterFloat paramLearningRate;

  SGDUpdater(Configuration const& config, VAMap const& parameters, VAMap const& gradients)
            : ParameterUpdater(parameters, gradients), learning_rate_(paramLearningRate(config)) {}
  ~SGDUpdater() {}

  virtual void update();
private:
  float learning_rate_;
};

class AdaDeltaUpdater : public ParameterUpdater {
public:
  static const ParameterFloat paramAdaDeltaMomentum;
  static const ParameterFloat paramLearningRate;

  AdaDeltaUpdater(Configuration const& config, VAMap const& parameters, VAMap const& gradients)
                 : ParameterUpdater(parameters, gradients), momentum_(paramAdaDeltaMomentum(config)),
                   learning_rate_(paramLearningRate(config)) {}
  ~AdaDeltaUpdater() {}

  virtual void update();
private:
  typedef std::map<std::string, std::valarray<float>> OwnedVAMap;

  float momentum_;
  float learning_rate_;

  OwnedVAMap gradient_rms_;
  OwnedVAMap update_rms_;

  std::valarray<float> update_buffer_;
};

class NnTrainer {
public:
  static const ParameterUInt   paramNumEpochs;
  static const ParameterUInt   paramStartEpoch;
  static const ParameterUInt   paramSeed;
  static const ParameterString paramUpdater;
  static const ParameterFloat  paramLearningRate;
  static const ParameterBool   paramRandomParamInit;
  static const ParameterString paramOutputDir;
  static const ParameterString paramNNTrainingStatsPath;

  NnTrainer(Configuration const& config, MiniBatchBuilder& mini_batch_builder, NeuralNetwork& nn);
  virtual ~NnTrainer();

  void train();
private:
  const size_t num_epochs_;
  const size_t start_epoch_;
  float        learning_rate_;
  bool         random_param_init_;
  std::string  output_dir_;
  std::string  nn_training_stats_path_;

  std::mt19937 rng_;

  MiniBatchBuilder& mini_batch_builder_;
  NeuralNetwork&    nn_;
  std::unique_ptr<ParameterUpdater> updater_;

  double compute_loss(std::valarray<float>  const& hyp, std::valarray<float>  const& ref, std::vector<unsigned> const& batch_mask,
                      size_t max_frames, size_t batch_size, size_t num_classes) const;
};

// -------------------- inline functions --------------------

inline size_t MiniBatchBuilder::batch_size() const {
  return batch_size_;
}

inline size_t MiniBatchBuilder::num_classes() const {
  return num_classes_;
}

inline size_t MiniBatchBuilder::max_seq_length() const {
  return max_seq_length_;
}

inline size_t MiniBatchBuilder::context_frames() const {
  return context_frames_;
}

inline size_t MiniBatchBuilder::feature_size() const {
  return base_feature_size_ * (2ul * context_frames_ + 1ul);
}

inline size_t MiniBatchBuilder::num_train_batches() const {
  return num_train_batches_;
}

inline size_t MiniBatchBuilder::num_cv_batches() const {
  return num_cv_batches_;
}

#endif /* __NN_TRAINING_HPP__ */
