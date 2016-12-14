/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>

#include "Recognizer.hpp"
#include "TdpModel.hpp"
#include "Timer.hpp"

/*****************************************************************************/

namespace {
}

/*****************************************************************************/

const ParameterDouble Recognizer::paramAmThreshold   ("am-threshold",   20.0);
const ParameterDouble Recognizer::paramWordPenalty   ("word-penalty",   10.0);

/*****************************************************************************/

void Recognizer::recognize(Corpus const& corpus) {
  EDAccumulator acc;
  size_t ref_total = 0ul;
  size_t sentence_errors = 0ul;
  std::vector<WordIdx> recognized_words;

  for (SegmentIdx s = 0ul; s < corpus.get_corpus_size(); s++) {
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    std::pair<WordIter, WordIter> ref_seq = corpus.get_word_sequence(s);

    recognizeSequence(features.first, features.second, recognized_words);

    EDAccumulator ed = editDistance(ref_seq.first, ref_seq.second, recognized_words.begin(), recognized_words.end());
    acc += ed;
    ref_total += std::distance(ref_seq.first, ref_seq.second);
    if (ed.total_count > 0u) {
      sentence_errors++;
    }

    const double wer = (static_cast<double>(acc.total_count) / static_cast<double>(ref_total)) * 100.0;
    std::cerr << (s + 1ul) << "/" << corpus.get_corpus_size()
              << " WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
              << "% (S/I/D) " << ed.substitute_count << "/" << ed.insert_count << "/" << ed.delete_count << " | ";

    std::copy(recognized_words.begin(), recognized_words.end(), std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << "| ";
    std::copy(ref_seq.first,            ref_seq.second,         std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << std::endl;
  }

  const double wer = (static_cast<double>(acc.total_count) / static_cast<double>(ref_total)) * 100.0;
  const double ser = (static_cast<double>(sentence_errors) / static_cast<double>(corpus.get_corpus_size())) * 100;
  const double time = 0.0; // TODO: compute
  const double rtf = 0.0; // TODO: compute

  std::cerr << "WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
            << "% (S/I/D) " << acc.substitute_count << "/" << acc.insert_count << "/" << acc.delete_count << std::endl;
  std::cerr << "SER: " << std::setw(6) << std::fixed << ser << "%" << std::endl;
  std::cerr << "Time: " << time << " seconds" << std::endl;
  std::cerr << "RTF: " << rtf << std::endl;
}

/*****************************************************************************/

void Recognizer::recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {
  // TODO: implement
}

/*****************************************************************************/

EDAccumulator Recognizer::editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end) {
  EDAccumulator result;
  size_t ref_size = ref_end - ref_begin;
  size_t hyp_size = rec_end - rec_begin;

  // Initialize the vectors containing the accumulators
  // with the dynamic programming initialization equations
  std::vector<EDAccumulator> current_rates(1+ref_size, EDAccumulator());
  for (size_t ref_idx = 1; ref_idx < ref_size; ref_idx++) {
  	current_rates[ref_idx] = current_rates[ref_idx-1];
  	current_rates[ref_idx].deletion_error();
  }
  std::vector<EDAccumulator> previous_rates (1+ref_size, EDAccumulator());

  for (size_t hyp_idx = 1; hyp_idx < hyp_size; hyp_idx++) {
  	// reset accumulators
  	current_rates.swap(previous_rates);

  	// add an insertion error for the very first accumulator (DP initialization)
  	current_rates[0].insertion_error();

  	// parse through all reference indices
  	for (size_t ref_idx = 1; ref_idx < ref_size; ref_idx++) {
  		uint16_t best_count = 0xFFFF;

  		// Words are equal -> move diagonally (with no error)
  		if (previous_rates[ref_idx-1].total_count < best_count && *(ref_begin+ref_idx) == *(rec_begin+hyp_idx)) {
  			current_rates[ref_idx] = previous_rates[ref_idx-1];
  		}

  		// Move diagonally and account for a substitution error
  		if (previous_rates[ref_idx-1].total_count + 1 < best_count) {
  			current_rates[ref_idx] = previous_rates[ref_idx-1];
  			current_rates[ref_idx].substitution_error();
  		}

  		// Move vertically and account for an insertion error
  		if (previous_rates[ref_idx].total_count + 1 < best_count) {
  			current_rates[ref_idx] = previous_rates[ref_idx];
  			current_rates[ref_idx].insertion_error();
  		}

  		// Move horizontally and account for an deletion error
  		if (current_rates[ref_idx-1].total_count + 1 < best_count) {
  			current_rates[ref_idx] = current_rates[ref_idx-1];
  			current_rates[ref_idx].deletion_error();
  		}
  	}
  }

  // In this position we have calculated the optimal distance between the two word sequences
  result = current_rates[ref_size];

  return result;
}

/*****************************************************************************/
