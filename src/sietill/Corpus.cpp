/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <sstream>
#include <utility>

#include "Corpus.hpp"
#include "IO.hpp"

// -------------------- CorpusDescription --------------------

const ParameterString CorpusDescription::paramSegmentName       ("name",    "");
const ParameterString CorpusDescription::paramSegmentSpeaker    ("speaker", "");
const ParameterString CorpusDescription::paramSegmentGender     ("gender",  "");
const ParameterString CorpusDescription::paramSegmentOrthography("orth",    "");

const ParameterString CorpusDescription::paramCorpusPath("corpus", "");

CorpusDescription::CorpusDescription(Configuration const& config) : path_(paramCorpusPath(config)) {
}

CorpusDescription::~CorpusDescription() {
}

void CorpusDescription::read(Lexicon const& lexicon) {
  Configuration corpus_config(path_);

  if (not corpus_config.is_array("segments")) {
    std::cerr << "Empty corpus" << std::endl;
    return;
  }

  std::vector<Configuration> segment_configs = corpus_config.get_array("segments");
  segments_.reserve(segment_configs.size());

  for (Configuration const& c : segment_configs) {
    segments_.push_back(read_segment(c, lexicon));
  }
}

CorpusDescription::segment_iterator CorpusDescription::begin() const {
  return segments_.begin();
}

CorpusDescription::segment_iterator CorpusDescription::end() const {
  return segments_.end();
}

Segment CorpusDescription::read_segment(Configuration const& config, Lexicon const& lexicon) {
  Segment result;

  result.name = paramSegmentName(config);
  std::string speaker = paramSegmentSpeaker(config);

  auto iter = speaker_map_.find(speaker);
  if (iter == speaker_map_.end()) {
    result.speaker = speaker_map_.size();
    speaker_map_[speaker] = result.speaker;
  }
  else {
    result.speaker = iter->second;
  }

  std::string gender = paramSegmentGender(config);
  iter = gender_map_.find(gender);
  if (iter == gender_map_.end()) {
    result.gender = gender_map_.size();
    gender_map_[gender] = result.gender;
  }
  else {
    result.gender = iter->second;
  }

  std::string orth = paramSegmentOrthography(config);
  std::stringstream ss(orth);
  std::string single;
  while (not (ss >> single).fail()) {
    result.orth.push_back(lexicon[single]);
  }

  return result;
}

// -------------------- Corpus --------------------

void Corpus::read(CorpusDescription const& corpus_description, std::string const& feature_path, SignalAnalysis& analyzer) {
  std::vector< std::vector<float> > buffer;
  features_per_timeframe_ = analyzer.n_features_total;
  frame_duration_         = static_cast<double>(analyzer.window_shift) / (analyzer.sample_rate * 1000.0);
  for (auto segment_iter = corpus_description.begin(); segment_iter != corpus_description.end(); ++segment_iter) {
    std::string file = feature_path + segment_iter->name + std::string(".mm2");
    files_.push_back(file);

    orths_.insert(orths_.end(), segment_iter->orth.begin(), segment_iter->orth.end());
    orth_offsets_.push_back(orths_.size());

    buffer.push_back(read_feature_file(file));
    analyzer.process_features(buffer.back());
    feature_offsets_.push_back(feature_offsets_.back() + buffer.back().size());
    std::cerr << '.';
  }
  std::cerr << std::endl;

  features_.resize(feature_offsets_.back());
  for (size_t i = 0ul; i < buffer.size(); i++) {
    std::copy(buffer[i].begin(), buffer[i].end(), features_.begin() + feature_offsets_[i]);
  }
}

size_t Corpus::get_corpus_size() const {
  return orth_offsets_.size() - 1u;
}

size_t Corpus::get_total_frame_count() const {
  return feature_offsets_.back();
}

size_t Corpus::get_max_seq_length() const {
  size_t max_len = 0ul;
  for (size_t s = 1ul; s < feature_offsets_.size(); s++) {
    max_len = std::max(max_len, (feature_offsets_[s] - feature_offsets_[s-1ul]) / features_per_timeframe_);
  }
  return max_len;
}

size_t Corpus::get_features_per_timeframe() const {
  return features_per_timeframe_;
}

double Corpus::get_frame_duration() const {
  return frame_duration_;
}

std::pair<WordIter, WordIter> Corpus::get_word_sequence(SegmentIdx idx) const {
  return std::make_pair(orths_.begin() + orth_offsets_[idx], orths_.begin() + orth_offsets_[idx + 1u]);
}

std::pair<FeatureIter, FeatureIter> Corpus::get_feature_sequence(SegmentIdx idx) const {
  return std::make_pair(FeatureIter(&features_[feature_offsets_[idx     ]], features_per_timeframe_),
                        FeatureIter(&features_[feature_offsets_[idx + 1u]], features_per_timeframe_));
}

std::pair<FeatureIter, FeatureIter> Corpus::get_all_features() const {
  return std::make_pair(FeatureIter(&features_[0ul],              features_per_timeframe_),
                        FeatureIter(&features_[features_.size()], features_per_timeframe_));
}

std::pair<size_t, size_t> Corpus::get_feature_offsets(SegmentIdx idx) const {
  return std::make_pair(feature_offsets_[idx], feature_offsets_[idx+1ul]);
}

std::string const& Corpus::get_file_name(SegmentIdx idx) const {
  return files_[idx];
}

