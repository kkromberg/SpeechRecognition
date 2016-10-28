/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>

#include "Timer.hpp"
#include "Training.hpp"

namespace {
}

/*****************************************************************************/

const ParameterUInt Trainer::paramMinObs      ("min-obs",        1u);
const ParameterUInt Trainer::paramNumSplits   ("num-splits",     1u);
const ParameterUInt Trainer::paramNumAligns   ("num-aligns",     1u);
const ParameterUInt Trainer::paramNumEstimates("num-estimates",  1u);
const ParameterUInt Trainer::paramNumMaxAligns("num-max-aligns", 1u);

const ParameterDouble Trainer::paramPruningThreshold   ("pruning-threshold",      50.0);

const ParameterString Trainer::paramMixturePath      ("mixture-path",        "");
const ParameterString Trainer::paramAlignmentPath    ("alignment-path",      "");
const ParameterString Trainer::paramTrainingStatsPath("training-stats-path", "");

const ParameterBool Trainer::paramWriteLinearSegmentation("write-linear-segmentation", false);
const ParameterBool Trainer::paramRealign                ("realign",                   true);
const ParameterBool Trainer::paramAlignmentPruning       ("alignment-pruning",         true);

/*****************************************************************************/

void Trainer::train(Corpus const& corpus) {
  Timer training_timer;
  Timer align_timer;
  Timer estimate_timer;
  Timer io_timer;
  Timer score_timer;

  training_timer.tick();

  size_t total_frames = 0ul;
  double am_score;
  std::vector<MarkovAutomaton> segment_automata;
  std::vector<size_t>          segment_offsets;
  Alignment                    alignment;

  std::ofstream stats_out;
  if (training_stats_path_.size() > 0ul) {
    stats_out.open(training_stats_path_, std::ios_base::out | std::ios_base::trunc);
    if (not stats_out.good()) {
      std::cerr << "Error opening stat file: " << training_stats_path_ << std::endl;
      abort();
    }
  }

  segment_offsets.push_back(0u);

  for (SegmentIdx s = 0u; s < corpus.get_corpus_size(); s++) {
    std::pair<WordIter, WordIter> segment = corpus.get_word_sequence(s);
    segment_automata.push_back(build_segment_automaton(segment.first, segment.second));
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    total_frames += features.second - features.first;
    segment_offsets.push_back(total_frames);
  }
  alignment.resize(total_frames * num_max_aligns_);

  for (SegmentIdx s = 0u; s < corpus.get_corpus_size(); s++) {
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    align_timer.tick();
    std::pair<size_t, size_t> boundaries = linear_segmentation(
        segment_automata[s],
        features.first,
        features.second,
        AlignmentIter(&*(alignment.begin() + segment_offsets[s     ] * num_max_aligns_), num_max_aligns_),
        AlignmentIter(&*(alignment.begin() + segment_offsets[s + 1u] * num_max_aligns_), num_max_aligns_)
    );
    align_timer.tock();
    if (write_linear_segmentation_) {
      io_timer.tick();
      write_linear_segmentation(corpus.get_file_name(s), boundaries.first, boundaries.second, features.first, features.second);
      io_timer.tock();
    }
  }

  std::cerr << "Linear alignment took " << align_timer.secs() << " seconds" << std::endl;
  align_timer.reset();

  std::cerr << "Acoustic Model Training" << std::endl;

  std::pair<FeatureIter, FeatureIter> features = corpus.get_all_features();
  estimate_timer.tick();
  mixtures_.accumulate(ConstAlignmentIter(&*alignment.begin(), num_max_aligns_),
                       ConstAlignmentIter(&*alignment.end(),   num_max_aligns_),
                       features.first, features.second, true);
  mixtures_.finalize();
  estimate_timer.tock();

  score_timer.tick();
  am_score = calc_am_score(corpus, alignment);
  score_timer.tock();
  std::cerr << "AM score: " << am_score << std::endl;
  if (training_stats_path_.size() > 0ul) {
    stats_out << "-1 0 0 " << am_score << std::endl;
  }
  std::cerr << "Num densities: " << mixtures_.num_densities() << std::endl;

  std::stringstream ss;
  ss << mixture_path_ << "lin.mix";
  io_timer.tick();
  std::ofstream mix_out(ss.str().c_str(), std::ios_base::out | std::ios_base::trunc);
  mixtures_.write(mix_out);
  io_timer.tock();

  for (size_t i = 0ul; i <= num_splits_; i++) {
    ConstAlignmentIter alignment_begin(&*alignment.begin(), num_max_aligns_);
    ConstAlignmentIter alignment_end  (&*alignment.end(),   num_max_aligns_);

    if (i > 0ul) {
      estimate_timer.tick();
      mixtures_.split(2u * min_obs_);
      mixtures_.accumulate(alignment_begin, alignment_end, features.first, features.second, false, max_approx_);
      mixtures_.finalize();

      mixtures_.eliminate(min_obs_);
      mixtures_.accumulate(alignment_begin, alignment_end, features.first, features.second, false, max_approx_);
      mixtures_.finalize();
      estimate_timer.tock();

      std::cerr << "Num densities: " << mixtures_.num_densities() << std::endl;
      score_timer.tick();
      am_score = calc_am_score(corpus, alignment);
      score_timer.tock();
      std::cerr << "AM score (post split): " << am_score << std::endl;
      if (training_stats_path_.size() > 0ul) {
        stats_out << i << " -1 0 " << am_score << std::endl;
      }
    }

    for (size_t j = 0ul; j < num_aligns_; j++) {
      if (realign_) {
        // realignment
        std::cerr << "Computing alignment" << std::endl;
        align_timer.tick();
        for (SegmentIdx s = 0ul; s < corpus.get_corpus_size(); s++) {
          std::pair<FeatureIter, FeatureIter> seq = corpus.get_feature_sequence(s);
          AlignmentIter alignment_begin(&*(alignment.begin() + segment_offsets[s]     * num_max_aligns_), num_max_aligns_);
          AlignmentIter alignment_end  (&*(alignment.begin() + segment_offsets[s+1ul] * num_max_aligns_), num_max_aligns_);
          if (alignment_pruning_) {
            aligner_.align_sequence_pruned(seq.first, seq.second,
                                           segment_automata[s],
                                           alignment_begin, alignment_end,
                                           pruning_threshold_);
          }
          else {
            aligner_.align_sequence_full(seq.first, seq.second,
                                         segment_automata[s],
                                         alignment_begin, alignment_end);
          }
        }
        align_timer.tock();

        if (alignment_path_.size() > 0ul) {
          io_timer.tick();
          std::stringstream aout;
          aout << alignment_path_ << i << "-" << j << ".dump";
          std::ofstream aoutstream(aout.str().c_str(), std::ios_base::out | std::ios_base::trunc);
          //dump_alignment(aoutstream, alignment, num_max_aligns_);
          write_alignment(aoutstream, alignment, num_max_aligns_);
          io_timer.tock();
        }
      }

      const size_t num_estimates = (i == 0ul) ? 1 : num_estimates_; // for single mixtures nothing changes by reestimating
      for (size_t k = 0ul; k < num_estimates; k++) {
        estimate_timer.tick();
        mixtures_.accumulate(alignment_begin, alignment_end, features.first, features.second, false, max_approx_);
        mixtures_.finalize();
        estimate_timer.tock();

        score_timer.tick();
        am_score = calc_am_score(corpus, alignment);
        score_timer.tock();
        if (training_stats_path_.size() > 0ul) {
          stats_out << i << " " << j << " " << k << " " << am_score << std::endl;
        }
        std::cerr << "AM score (accumulate): " << am_score << std::endl;
      }
    }

    io_timer.tick();
    std::stringstream ss;
    ss << mixture_path_ << i << ".mix";
    std::ofstream mix_out(ss.str().c_str(), std::ios_base::out | std::ios_base::trunc);
    if (not mix_out.good()) {
      std::cerr << "Could not open " << ss.str() << std::endl;
    }
    else {
      mixtures_.write(mix_out);
    }

    io_timer.tock();
  }

  training_timer.tock();

  std::cerr << "Estimation  took " << estimate_timer.secs() << " seconds" << std::endl;
  std::cerr << "Alignment   took " << align_timer.secs()    << " seconds" << std::endl;
  std::cerr << "IO          took " << io_timer.secs()       << " seconds" << std::endl;
  std::cerr << "Score comp. took " << score_timer.secs()    << " seconds" << std::endl;
  std::cerr << "Training    took " << training_timer.secs() << " seconds" << std::endl;
}

/*****************************************************************************/

MarkovAutomaton Trainer::build_segment_automaton(WordIter segment_begin, WordIter segment_end) const {
  //TODO: implement
  return MarkovAutomaton();
}

/*****************************************************************************/

std::pair<size_t, size_t> Trainer::linear_segmentation(MarkovAutomaton const& automaton,
                                                       FeatureIter   feature_begin, FeatureIter   feature_end,
                                                       AlignmentIter align_begin,   AlignmentIter align_end) const {
  std::pair<size_t, size_t> boundaries;
  //TODO: implement
  return boundaries;
}

/*****************************************************************************/

void Trainer::write_linear_segmentation(std::string const& feature_path,
                                        size_t speech_begin, size_t speech_end,
                                        FeatureIter feature_begin, FeatureIter feature_end) const {
  std::string output_path = feature_path.substr(0, feature_path.size() - 4) + ".seg";

  std::ofstream out(output_path.c_str());
  if (!out.good()) {
    std::cerr << "Error opening " << output_path << std::endl;
    return;
  }

  size_t idx = 0u;
  while (feature_begin != feature_end) {
    out << idx << " " << **feature_begin << std::endl;
    ++idx;
    ++feature_begin;
  }

  out << std::endl << speech_begin    << " -0.1 " << std::endl << speech_begin   << " .15" << std::endl;
  out << std::endl << speech_end - 1u << " -0.1 " << std::endl << speech_end -1u << " .15" << std::endl;
}

/*****************************************************************************/

double Trainer::calc_am_score(Corpus const& corpus, Alignment const& alignment) const {
  // TODO: implement
  return 0.0;
}

/*****************************************************************************/
