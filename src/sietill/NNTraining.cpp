/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "NNTraining.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Alignment.hpp"
#include "Timer.hpp"
#include "Util.hpp"

namespace {
  ParameterUpdater* get_updater(std::string const& name, Configuration const& config,
                                ParameterUpdater::VAMap const& parameters, ParameterUpdater::VAMap const& gradients) {
    if (name == "sgd") {
      return new SGDUpdater(config, parameters, gradients);
    }
    else if (name == "adadelta") {
      return new AdaDeltaUpdater(config, parameters, gradients);
    }
    else {
      std::cerr << "Unknown updater: " << name << std::endl;
      abort();
    }
  }
}

const ParameterUInt   MiniBatchBuilder::paramContextFrames            ("context-frames", 0u);
const ParameterUInt   MiniBatchBuilder::paramSeed                     ("seed", 0x58DBFDD0);
const ParameterUInt   MiniBatchBuilder::paramMaxSilenceFrames         ("max-silence-frames", 0xFFFFFFFF);
const ParameterBool   MiniBatchBuilder::paramNormalizeFeaturesPerBatch("normalize-features-per-batch", false);
const ParameterString MiniBatchBuilder::paramTargetFile               ("target-file", "");
const ParameterFloat  MiniBatchBuilder::paramCVSize                   ("cv-size", 0.0f);

MiniBatchBuilder::MiniBatchBuilder(Configuration const& config, Corpus const& corpus, size_t batch_size, size_t num_classes, StateIdx silence_state)
                                  : batch_size_(batch_size),
                                    num_classes_(num_classes),
                                    max_seq_length_(corpus.get_max_seq_length()),
                                    base_feature_size_(corpus.get_features_per_timeframe()),
                                    context_frames_(paramContextFrames(config)),
                                    silence_state_(silence_state),
                                    max_silence_frames_(paramMaxSilenceFrames(config)),
                                    normalize_features_per_batch_(paramNormalizeFeaturesPerBatch(config)),
                                    target_file_(paramTargetFile(config)),
                                    cv_size_(paramCVSize(config)),
                                    num_train_seq_(static_cast<size_t>(corpus.get_corpus_size() * (1.0f - cv_size_))),
                                    num_cv_seq_(corpus.get_corpus_size() - num_train_seq_),
                                    num_train_batches_((num_train_seq_ + batch_size_ - 1ul) / batch_size_),
                                    num_cv_batches_   ((num_cv_seq_    + batch_size_ - 1ul) / batch_size_),
                                    corpus_(corpus),
                                    rng_(paramSeed(config)) {
  train_segment_sequence_.resize(corpus.get_corpus_size());
  cv_segment_sequence_.resize(num_cv_seq_);
  std::iota(train_segment_sequence_.begin(), train_segment_sequence_.end(), 0ul);
  shuffle();
  std::copy(train_segment_sequence_.begin() + num_train_seq_, train_segment_sequence_.end(), cv_segment_sequence_.begin());
  train_segment_sequence_.resize(num_train_seq_);

  std::ifstream in(target_file_);
  if (!in.good()) {
    std::cerr << "Error reading target alignment: " << target_file_ << std::endl;
    std::abort();
  }
  read_alignment(in, alignment_, max_aligns_);

  if (normalize_features_per_batch_) {
    mean_.resize     (feature_size());
    next_mean_.resize(feature_size());
    sqr_.resize      (feature_size());
    diff_.resize     (feature_size());
    diff2_.resize    (feature_size());
  }
}

MiniBatchBuilder::~MiniBatchBuilder() {
}

void MiniBatchBuilder::shuffle() {
  for (size_t s = 0ul; s < train_segment_sequence_.size(); s++) {
    std::uniform_int_distribution<size_t> dist(s, train_segment_sequence_.size() - 1ul);
    std::swap(train_segment_sequence_[s], train_segment_sequence_[dist(rng_)]);
  }
}

void MiniBatchBuilder::build_batch(size_t batch_index, bool cv, std::valarray<float>& output, std::gslice const& slice,
                                   std::valarray<float>& targets, std::vector<unsigned>& mask) const {
  mask.resize(batch_size_);
  targets.resize(max_seq_length_ * batch_size_ * num_classes_);
  targets = 0.0f;

  std::fill(mask.begin(), mask.end(), 0u);
  targets = 0.0f;
  output[slice] = 0.0f;

  const size_t segment_size = cv ? cv_segment_sequence_.size() : train_segment_sequence_.size();
  size_t max_length = 0ul;
  size_t s, i;
  for (s = batch_index * batch_size_, i = 0ul;
       s < (batch_index + 1ul) * batch_size_ and s < segment_size;
       s++,i++) {
    const size_t segment = cv ? cv_segment_sequence_[s] : train_segment_sequence_[s];
    std::pair<FeatureIter, FeatureIter> iters = corpus_.get_feature_sequence(segment);

    std::pair<size_t, size_t> offsets = corpus_.get_feature_offsets(segment);
    offsets.first /= iters.first.size;
    offsets.second /= iters.second.size;

    std::pair<size_t, size_t> trunc_offsets = sequence_boundaries(offsets.first, offsets.second);
    trunc_offsets.second = trunc_offsets.first + std::min<size_t>(trunc_offsets.second - trunc_offsets.first, slice.size()[0]);

    mask[i]    = trunc_offsets.second - trunc_offsets.first;
    max_length = std::max<size_t>(max_length, mask[i]);

    size_t center_time = 0ul;
    while (iters.first != iters.second) {
      for (int delta = -context_frames_; delta <= static_cast<int>(context_frames_); delta++) {
        const int feature_time = static_cast<int>(center_time) + delta;
        if (feature_time < static_cast<int>(trunc_offsets.first) or feature_time >= static_cast<int>(trunc_offsets.second)) {
          continue;
        }
        for (size_t f = 0ul; f < iters.first.size; f++) {
          size_t index =   slice.start()
                         + (center_time + delta - trunc_offsets.first)          * slice.stride()[0]
                         + i                                                    * slice.stride()[1]
                         + ((delta + context_frames_) * base_feature_size_ + f) * slice.stride()[2];
          float val = (*iters.first)[f];
          output[index] = val;
        }
      }

      if (center_time >= trunc_offsets.first and center_time < trunc_offsets.second) {
        for (size_t a = 0ul; a < alignment_[max_aligns_ * (offsets.first + trunc_offsets.first + center_time)].count; a++) {
          AlignmentItem const& aitem = alignment_[max_aligns_ * (offsets.first + center_time) + a];
          targets[(center_time - trunc_offsets.first) * batch_size_ * num_classes_ + i * num_classes_ + aitem.state] = aitem.weight;
        }
      }
      ++iters.first;
      ++center_time;
    }
  }

  if (normalize_features_per_batch_) {
    mean_ = 0.0f;
    sqr_  = 0.0f;

    size_t k = 0ul;
    for (size_t t = 0ul; t < max_length; t++) {
      for (size_t b = 0ul; b < batch_size_; b++) {
        if (t >= mask[b]) {
          continue;
        }
        k += 1ul;
        std::slice frame_slice(t * slice.stride()[0] + b * slice.stride()[1], slice.size()[2], slice.stride()[2]);

        //S = S + (x - M)*(x - Mnext)
        diff_  = output[frame_slice];
        diff2_ = diff_;
        diff_ -= mean_;
        next_mean_ = mean_ + diff_ / static_cast<float>(k);
        diff2_ -= next_mean_;
        sqr_ += diff_ * diff2_;
        mean_ = next_mean_;
      }
    }
    sqr_ = 1.0f / std::sqrt(sqr_ / static_cast<float>(k - 1ul));

    for (size_t t = 0ul; t < max_length; t++) {
      for (size_t b = 0ul; b < batch_size_; b++) {
        if (t >= mask[b]) {
          continue;
        }
        std::slice frame_slice(t * slice.stride()[0] + b * slice.stride()[1], slice.size()[2], slice.stride()[2]);
        output[frame_slice] -= mean_;
        output[frame_slice] *= sqr_;
      }
    }
  }
}

std::pair<size_t, size_t> MiniBatchBuilder::sequence_boundaries(size_t begin, size_t end) const {
  size_t initial_silence_frames = 0ul;
  while (alignment_[max_aligns_ * (begin + initial_silence_frames)].state == silence_state_) {
    initial_silence_frames += 1ul;
  }

  size_t final_silence_frames = 0ul;
  while (alignment_[max_aligns_ * (end - final_silence_frames)].state == silence_state_) {
    final_silence_frames += 1ul;
  }

  return std::make_pair<size_t, size_t>(std::max(initial_silence_frames, max_silence_frames_) - max_silence_frames_,
                                        end - begin - std::max(final_silence_frames, max_silence_frames_) + max_silence_frames_);
}

// -------------------- ParameterUpdater  --------------------

const ParameterFloat SGDUpdater::paramLearningRate("learning-rate", 0.001);

void SGDUpdater::update() {
  // TODO: implement
}

const ParameterFloat AdaDeltaUpdater::paramAdaDeltaMomentum("adadelta-momentum", 0.90);
const ParameterFloat AdaDeltaUpdater::paramLearningRate("learning-rate", 0.001);

void AdaDeltaUpdater::update() {
  // TODO: implement
}

// -------------------- NnTrainer  --------------------

const ParameterUInt   NnTrainer::paramNumEpochs          ("num-epochs",             1u);
const ParameterUInt   NnTrainer::paramStartEpoch         ("start-epoch",            1u);
const ParameterUInt   NnTrainer::paramSeed               ("param-init-seed",        498061416);
const ParameterString NnTrainer::paramUpdater            ("updater",                "sgd");
const ParameterFloat  NnTrainer::paramLearningRate       ("learning-rate",          0.001);
const ParameterBool   NnTrainer::paramRandomParamInit    ("random-param-init",      true);
const ParameterString NnTrainer::paramOutputDir          ("output-dir",             "./models");
const ParameterString NnTrainer::paramNNTrainingStatsPath("nn-training-stats-path", "");

NnTrainer::NnTrainer(Configuration const& config, MiniBatchBuilder& mini_batch_builder, NeuralNetwork& nn)
                           : num_epochs_(paramNumEpochs(config)), start_epoch_(std::max(1u, paramStartEpoch(config))),
                             learning_rate_(paramLearningRate(config)), random_param_init_(paramRandomParamInit(config)),
                             output_dir_(paramOutputDir(config)), nn_training_stats_path_(paramNNTrainingStatsPath(config)),
                             rng_(paramSeed(config)), mini_batch_builder_(mini_batch_builder), nn_(nn),
                             updater_(get_updater(paramUpdater(config), config, nn_.get_parameters(), nn_.get_gradients())) {
}

NnTrainer::~NnTrainer() {
}


void NnTrainer::train() {
  const size_t num_classes = mini_batch_builder_.num_classes();
  const size_t batch_size = mini_batch_builder_.batch_size();
  std::normal_distribution<float> dist(0.0, 0.1);
  nn_.init_parameters([&](){ return dist(rng_); });

  if (start_epoch_ > 1ul) {
    std::stringstream ss;
    ss << output_dir_ << '/' << (start_epoch_ - 1ul) << '/';
    nn_.load(ss.str());
  }

  std::vector<unsigned> mask;
  std::valarray<float> targets;
  for (size_t epoch = start_epoch_; epoch <= num_epochs_; epoch++) {
    size_t total_frames           = 0ul;
    size_t total_incorrect_frames = 0ul;
    mini_batch_builder_.shuffle();

    for (size_t batch = 0ul; batch < mini_batch_builder_.num_train_batches(); batch++) {
      Timer batch_timer;
      batch_timer.tick();

      nn_.get_feature_buffer() = 0.0f;
      mini_batch_builder_.build_batch(batch, false,
                                      nn_.get_feature_buffer(),
                                      nn_.get_feature_buffer_slice(),
                                      targets,
                                      nn_.get_batch_mask());
      nn_.forward();

      size_t max_len = *std::max_element(nn_.get_batch_mask().begin(), nn_.get_batch_mask().end());
      size_t batch_frames     = 0ul;
      size_t incorrect_frames = 0ul;
      for (size_t t = 0ul; t < max_len; t++) {
        for (size_t b = 0ul; b < batch_size; b++) {
          if (t < nn_.get_batch_mask()[b]) {
            float const* score_begin = &nn_.get_score_buffer()[t * batch_size * num_classes + b * num_classes];
            float const* max_score   = std::max_element(score_begin, score_begin + num_classes);
            size_t       hyp_class   = std::distance(score_begin, max_score);

            float const* ref_begin = &targets[t * batch_size * num_classes + b * num_classes];
            float const* max_ref   = std::max_element(ref_begin, ref_begin + num_classes);
            size_t       ref_class = std::distance(ref_begin, max_ref);

            batch_frames += 1ul;
            if (hyp_class != ref_class) {
              incorrect_frames += 1ul;
            }
          }
        }
      }
      double batch_loss = compute_loss(nn_.get_score_buffer(), targets, nn_.get_batch_mask(), max_len, batch_size, num_classes);

      nn_.backward(targets);
      updater_->update();

      total_frames           += batch_frames;
      total_incorrect_frames += incorrect_frames;

      batch_timer.tock();

      std::cerr << "epoch: " << epoch << "/" << num_epochs_
                << " batch: " << (batch + 1ul) << "/" << mini_batch_builder_.num_train_batches()
                << std::fixed << std::setprecision(6)
                << " | loss: " << batch_loss
                << " | time: " << batch_timer.secs() << std::endl;
    }

    size_t cv_total_frames = 0ul;
    size_t cv_errors       = 0ul;
    for (size_t batch = 0ul; batch < mini_batch_builder_.num_cv_batches(); batch++) {
      nn_.get_feature_buffer() = 0.0f;
      mini_batch_builder_.build_batch(batch, false,
                                      nn_.get_feature_buffer(),
                                      nn_.get_feature_buffer_slice(),
                                      targets,
                                      nn_.get_batch_mask());
      nn_.forward();

      size_t max_len = *std::max_element(nn_.get_batch_mask().begin(), nn_.get_batch_mask().end());
      for (size_t t = 0ul; t < max_len; t++) {
        for (size_t b = 0ul; b < batch_size; b++) {
          if (t < nn_.get_batch_mask()[b]) {
            float const* score_begin = &nn_.get_score_buffer()[t * batch_size * num_classes + b * num_classes];
            float const* max_score   = std::max_element(score_begin, score_begin + num_classes);
            size_t       hyp_class   = std::distance(score_begin, max_score);

            float const* ref_begin = &targets[t * batch_size * num_classes + b * num_classes];
            float const* max_ref   = std::max_element(ref_begin, ref_begin + num_classes);
            size_t       ref_class = std::distance(ref_begin, max_ref);

            cv_total_frames += 1ul;
            if (hyp_class != ref_class) {
              cv_errors += 1ul;
            }
          }
        }
      }
    }

    std::stringstream ss;
    ss << output_dir_ << '/' << epoch << '/';
    nn_.save(ss.str());
    std::cerr << "Epoch train frame error-rate: " << (static_cast<double>(total_incorrect_frames) / static_cast<double>(total_frames))    << std::endl;
    std::cerr << "Epoch cv    frame error-rate: " << (static_cast<double>(cv_errors)              / static_cast<double>(cv_total_frames)) << std::endl;
  }
}

double NnTrainer::compute_loss(std::valarray<float>  const& hyp, std::valarray<float>  const& ref, std::vector<unsigned> const& batch_mask,
                               size_t max_frames, size_t batch_size, size_t num_classes) const {
  double sum = 0.0;
  size_t num_frames = 0ul;
  for (size_t t = 0ul; t < max_frames; t++) {
    size_t t_idx = t * batch_size * num_classes;
    for (size_t b = 0ul; b < batch_size; b++) {
      if (t >= batch_mask[b]) {
        continue;
      }
      size_t b_idx = t_idx + b * num_classes;
      num_frames += 1ul;
      for (size_t f = 0ul; f < num_classes; f++) {
        double hyp_prob = hyp[b_idx + f];
        double ref_prob = ref[b_idx + f];
        if (ref_prob > 0.0 and hyp_prob > 0.0) {
          sum += ref_prob * std::log(hyp_prob);
        }
      }
    }
  }

  return -sum / num_frames;
}

