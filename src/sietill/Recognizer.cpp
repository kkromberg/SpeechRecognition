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
	size_t num_features = feature_end - feature_begin;
	size_t num_states 	= lexicon_.num_states();
	size_t num_words 		= lexicon_.num_words();
	size_t num_states_word[num_words];

	// save number of states for each word
	for (WordIdx word_idx = 0; word_idx < lexicon_.num_words(); word_idx++) { // loop words
		MarkovAutomaton current_automaton = lexicon_.get_automaton_for_word(word_idx);
		num_states_word[word_idx] = current_automaton.num_states();
	}
	Book book[num_features];
	Book hyp[num_words][num_states];
	Book hypTmp[num_states];

	// initialize hypothesis with infinity score
	for (WordIdx word_idx = 1; word_idx <= num_words; word_idx++) {
		for (StateIdx state_idx = 0; state_idx <= num_states; state_idx++) {
			hyp[word_idx][state_idx].score = std::numeric_limits<double>::infinity();
		}
	}


	// backpointer matrix
	BackpointerMatrix bp_matrix;
	for (StateIdx state_idx = 0; state_idx < num_states; state_idx++) {
		bp_matrix.push_back(std::vector<size_t>(num_features, -1));
	}
	double local_distance;

	// costs for virtual state
	book[0].score = 0.0;
	size_t frame_counter = 1;
	MarkovAutomaton current_automaton;
	double tmp_score;
	for (FeatureIter iter = feature_begin + 1; iter != feature_end; iter++, frame_counter++) { // loop features
		for (WordIdx word_idx = 0; word_idx <= num_words; word_idx++) { // loop words
			// word transition
			// Q(t-1,0; w) = Q(t-1, S(W(t-1)); W(t-1)) - log p(w)
			hyp[word_idx][0].score = book[frame_counter-1].score - 1; // -log p(w) = const -> zerogram
			hyp[word_idx][0].bkp 	 = frame_counter - 1;
			// B(t-1,0; w) = t-1
			current_automaton = lexicon_.get_automaton_for_word(word_idx);
			for (StateIdx state_idx = 1; state_idx <= current_automaton.num_states(); state_idx++) { // loop states
				hypTmp[state_idx].score = std::numeric_limits<double>::infinity();

				for (StateIdx pre = std::max(0, state_idx - 2); pre <= state_idx; pre++) { // loop over predecessor states
					tmp_score = hyp[word_idx][pre].score + tdp_model_.score(current_automaton[state_idx - pre], pre);

					if (tmp_score < hypTmp[state_idx].score) {
						hypTmp[state_idx].score = tmp_score;
						hypTmp[state_idx].bkp   = hyp[word_idx][pre].bkp;
					}
				}

				// store result into hypothesis
				for (StateIdx state_idx = 1; state_idx <= current_automaton.num_states(); state_idx++) {
					hyp[word_idx][state_idx].bkp 	 = hypTmp[state_idx].bkp;
					hyp[word_idx][state_idx].score = hypTmp[state_idx].score + scorer_.score(iter, current_automaton[state_idx]);
				}

				book[frame_counter].score = std::numeric_limits<double>::infinity();
				// -log p(x_t|\teta)
				local_distance = scorer_.score(iter, current_automaton[state_idx]);

			} // loop states
		} // loop words

		for (WordIdx word_idx = 1; word_idx <= num_words; word_idx++) {
			if (hyp[word_idx][num_states[word_idx]].score < book[feature_counter].score) {
				book[feature_counter].score = hyp[word_idx][num_states[word_idx]].score;
				book[feature_counter].bkp 	= hyp[word_idx][num_states[word_idx]].bkp;
				book[feature_counter].word  = word_idx;

			}
		}
	} // loop features

	// trace back
	size_t count = 1;
	size_t feature_index = feature_number - 1;

	while (feature_index > 0) {
		output[count] = book[feature_index].word;

		feature_index = book[feature_index].bkp;
		count++;
	}
}

/*****************************************************************************/

EDAccumulator Recognizer::editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end) {
  EDAccumulator result;
  // TODO: implement
  return result;
}

/*****************************************************************************/
