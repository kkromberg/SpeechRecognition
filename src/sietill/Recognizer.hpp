/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __RECOGNIZER_HPP__
#define __RECOGNIZER_HPP__

#include "Corpus.hpp"
#include "Lexicon.hpp"
#include "Mixtures.hpp"
#include "TdpModel.hpp"


#include <memory>

struct EDAccumulator {
  uint16_t total_count;
  uint16_t substitute_count;
  uint16_t insert_count;
  uint16_t delete_count;

  EDAccumulator() : total_count(0u), substitute_count(0u), insert_count(0u), delete_count(0u) {}
  EDAccumulator(uint16_t total_count, uint16_t substitute_count, uint16_t insert_count, uint16_t delete_count)
               : total_count(total_count), substitute_count(substitute_count), insert_count(insert_count), delete_count(delete_count) {}

  EDAccumulator operator+(EDAccumulator const& other) const {
    return EDAccumulator(total_count      + other.total_count,
                         substitute_count + other.substitute_count,
                         insert_count     + other.insert_count,
                         delete_count     + other.delete_count);
  }

  EDAccumulator& operator+=(EDAccumulator const& other) {
    total_count      += other.total_count;
    substitute_count += other.substitute_count;
    insert_count     += other.insert_count;
    delete_count     += other.delete_count;
    return *this;
  }

  void reset() { total_count = 0u; substitute_count = 0u; insert_count = 0u; delete_count = 0u; }
  void substitution_error() { total_count++; substitute_count++; }
  void insertion_error()    { total_count++; insert_count++;     }
  void deletion_error()     { total_count++; delete_count++;     }
};

struct Hypothesis;
typedef std::shared_ptr<Hypothesis> HypothesisPtr;

struct Hypothesis {
	HypothesisPtr ancestor_;
  double        score_;
  StateIdx      state_;
  WordIdx       word_;
  bool          new_word_;

  Hypothesis() :
    ancestor_(HypothesisPtr(nullptr)), score_(0.0), state_(0), word_(0), new_word_(false) {}

  Hypothesis(HypothesisPtr ancestor, double score, StateIdx state, WordIdx word, bool new_word) :
    ancestor_(ancestor), score_(score), state_(state), word_(word), new_word_(new_word){}

  Hypothesis(const Hypothesis &hyp) :
    ancestor_(hyp.ancestor_), score_(hyp.score_), state_(hyp.state_), word_(hyp.word_), new_word_(hyp.new_word_) {}

  bool is_initial() {
  	return ancestor_ == nullptr ? true : false;
  }

};


struct Book {
	// back trace information about best ending word for each frame
	 double score;
	 uint16_t word;
	 uint16_t bkp;
	 int state_idx;

	 Book(double score, uint16_t word, uint16_t bkp) : score(score), word(word), bkp(bkp), state_idx(-1) {}

	 Book(double score, uint16_t bkp) : score(score), word(-1), bkp(bkp), state_idx(-1) {}

	 Book() : score(0.0), word(0), bkp(0), state_idx(-1) {}
};

class Recognizer {
public:
  static const ParameterBool    paramLookahead;
  static const ParameterDouble  paramAmThreshold;
  static const ParameterDouble  paramLookaheadScale;
  static const ParameterDouble  paramWordPenalty;
  static const ParameterBool    paramPrunedSearch;
  static const ParameterInt     paramMaxRecognitionRuns;

	// Data structures for a 2D dynamic programming problem
	typedef std::vector< std::vector<size_t> > 	 BackpointerMatrix;
	typedef std::vector< std::vector<size_t> > 	 TracebackMatrix;

  Recognizer(Configuration const& config, Lexicon const& lexicon, FeatureScorer& scorer, TdpModel const& tdp_model)
            : am_threshold_(paramAmThreshold(config)),
              word_penalty_(paramWordPenalty(config)),
              pruned_search_(paramPrunedSearch(config)),
              max_recognition_runs_(paramMaxRecognitionRuns(config)),
              lexicon_(lexicon), scorer_(scorer), tdp_model_(tdp_model) {}

  void recognize(Corpus const& corpus);
  void recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output);
  void recognizeSequence_pruned(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output);


  EDAccumulator editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end);
private:
  const double am_threshold_;
  const double word_penalty_;
  const bool   pruned_search_;
  const size_t max_recognition_runs_;

  Lexicon  const& lexicon_;
  FeatureScorer&  scorer_;
  TdpModel const& tdp_model_;

  typedef std::vector<HypothesisPtr> Beam;
  typedef std::vector<HypothesisPtr>::iterator BeamIterator;
};

#endif /* __RECOGNIZER_HPP__ */
