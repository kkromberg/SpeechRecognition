/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __CORPUS_HPP__ 
#define __CORPUS_HPP__

#include <unordered_map>
#include <vector>

#include "Config.hpp"
#include "Iter.hpp"
#include "Lexicon.hpp"
#include "SignalAnalysis.hpp"
#include "Types.hpp"

struct Segment {
  std::string          name;
  size_t               speaker;
  size_t               gender;
  std::vector<WordIdx> orth;
};

class CorpusDescription {
public:
  typedef std::vector<Segment>::const_iterator segment_iterator;

  static const ParameterString paramCorpusPath;

  CorpusDescription(Configuration const& config);
  ~CorpusDescription();

  void read(Lexicon const& lexicon);

  segment_iterator begin() const;
  segment_iterator end() const;
private:
  static const ParameterString paramSegmentName;
  static const ParameterString paramSegmentSpeaker;
  static const ParameterString paramSegmentGender;
  static const ParameterString paramSegmentOrthography;

  std::string path_;
  std::vector<Segment> segments_;
  std::unordered_map<std::string, size_t> speaker_map_;
  std::unordered_map<std::string, size_t> gender_map_;

  Segment read_segment(Configuration const& config, Lexicon const& lexicon);
};

class Corpus {
public:
  Corpus() : features_per_timeframe_(0ul) {
    feature_offsets_.push_back(0u);
    orth_offsets_.push_back(0u);
  }
  ~Corpus() {
  }

  void read(CorpusDescription const& corpus_description, std::string const& feature_path, SignalAnalysis& analyzer);

  size_t                              get_corpus_size() const;
  size_t                              get_total_frame_count() const;
  size_t                              get_max_seq_length() const;
  size_t                              get_features_per_timeframe() const;
  double                              get_frame_duration() const;
  std::pair<WordIter, WordIter>       get_word_sequence(SegmentIdx idx) const;
  std::pair<FeatureIter, FeatureIter> get_feature_sequence(SegmentIdx idx) const;
  std::pair<FeatureIter, FeatureIter> get_all_features() const;
  std::pair<size_t, size_t>           get_feature_offsets(SegmentIdx idx) const;
  std::string const&                  get_file_name(SegmentIdx idx) const;
private:
  size_t                   features_per_timeframe_;
  double                   frame_duration_;
  std::vector<size_t>      feature_offsets_;
  std::vector<float>       features_;
  std::vector<size_t>      orth_offsets_;
  std::vector<WordIdx>     orths_;
  std::vector<std::string> files_;
};

#endif /* __CORPUS_HPP__ */
