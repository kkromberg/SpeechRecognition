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
#include <algorithm>    // std::move (ranges)
#include <utility>      // std::move (objects)
#include <sstream>
#include <fstream>

#include "Recognizer.hpp"
#include "TdpModel.hpp"
#include "Timer.hpp"
#include "Util.hpp"


/*****************************************************************************/

namespace {
}

/*****************************************************************************/

const ParameterDouble    Recognizer::paramAmThreshold          ("am-threshold" ,   20.0);
const ParameterDouble    Recognizer::paramWordPenalty          ("word-penalty" ,   10.0);
const ParameterBool      Recognizer::paramPrunedSearch         ("pruned-search",   true);
const ParameterInt       Recognizer::paramMaxRecognitionRuns   ("max-recognition-runs",   1000);

/*****************************************************************************/

void Recognizer::recognize(Corpus const& corpus) {
	Timer search_timer;
  EDAccumulator acc;
  size_t ref_total = 0ul;
  size_t sentence_errors = 0ul;
  size_t corpus_size = std::min(corpus.get_corpus_size(), max_recognition_runs_);

  search_timer.tick();
#pragma omp parallel for ordered schedule(dynamic)
  for (SegmentIdx s = 0ul; s < corpus_size; s++) {
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    std::pair<WordIter, WordIter> ref_seq = corpus.get_word_sequence(s);
    std::vector<WordIdx> recognized_words;

    if (pruned_search_) {
      recognizeSequence_pruned(features.first, features.second, recognized_words);
    } else {
      recognizeSequence(features.first, features.second, recognized_words);
    }

    EDAccumulator ed = editDistance(ref_seq.first, ref_seq.second, recognized_words.begin(), recognized_words.end());
    acc += ed;
    ref_total += std::distance(ref_seq.first, ref_seq.second);
    if (ed.total_count > 0u) {
      sentence_errors++;
    }

#pragma omp ordered
{
		search_timer.tock();
    const double wer = (static_cast<double>(ed.total_count) / static_cast<double>(ref_seq.second - ref_seq.first)) * 100.0;
    std::cerr << (s + 1ul) << "/" << corpus_size
              << " WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
              << "% (S/I/D) " << ed.substitute_count << "/" << ed.insert_count << "/" << ed.delete_count << " | ";

    std::copy(recognized_words.begin(), recognized_words.end(), std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << "| ";
    std::copy(ref_seq.first,            ref_seq.second,         std::ostream_iterator<WordIdx>(std::cerr, " "));
    std::cerr << std::endl;
    search_timer.tick();
}
  }
  search_timer.tock();

  const double wer = (static_cast<double>(acc.total_count) / static_cast<double>(ref_total)) * 100.0;
  const double ser = (static_cast<double>(sentence_errors) / static_cast<double>(corpus_size)) * 100;
  const double time = search_timer.secs(); // TODO: compute
  const double rtf = time / (corpus.get_frame_duration() * corpus.get_total_frame_count()) ; // TODO: compute

  std::cerr << "WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
            << "% (S/I/D) " << acc.substitute_count << "/" << acc.insert_count << "/" << acc.delete_count << std::endl;
  std::cerr << "SER: " << std::setw(6) << std::fixed << ser << "%" << std::endl;
  std::cerr << "Time: " << time << " seconds" << std::endl;
  std::cerr << "RTF: " << rtf << std::endl;
}

void Recognizer::merge_hypothesis(double score, size_t state_idx, size_t t, WordIdx word_idx, Book& target_hyp) {
	target_hyp.score = score;
	target_hyp.state_idx = state_idx;
	target_hyp.bkp = t;
	target_hyp.word = word_idx;
}

/*****************************************************************************/

void Recognizer::recognizeSequence_pruned(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {
	scorer_.prepare_sequence(feature_begin, feature_end);

	size_t n_words    = lexicon_.num_words();
	size_t n_states   = lexicon_.num_states();
	size_t n_features = feature_end - feature_begin;

	std::vector<size_t> n_states_per_word(n_words, 0);
	for (size_t w = 0; w < n_words; w++) {
		n_states_per_word[w] = lexicon_.get_automaton_for_word(w).num_states();
	}
	size_t max_states_per_word = *max_element(n_states_per_word.begin(), n_states_per_word.end());

	std::vector<Book> current_hyps(n_states * max_states_per_word, Book(std::numeric_limits<double>::infinity(), 0, 0, 0));
	std::vector<Book> next_hyps   (n_states * max_states_per_word, Book(std::numeric_limits<double>::infinity(), 0, 0, 0));
	std::vector<Book> traceback   (n_features + 1                , Book(0.0, 0, 0, 0));

	current_hyps[0].score = 0.0;

	int t = 1;
	std::vector<double> am_cache(n_states, std::numeric_limits<double>::infinity());
	for (FeatureIter cur_frame = feature_begin; cur_frame != feature_end; cur_frame++, t++) {
		double best_score = std::numeric_limits<double>::infinity();
		for (std::vector<Book>::iterator cur_hyp = current_hyps.begin(); cur_hyp != current_hyps.end(); cur_hyp++) {
			if (cur_hyp->score == std::numeric_limits<double>::infinity()) {
				continue;
			}

			if ((size_t) cur_hyp->state_idx == n_states_per_word[cur_hyp->word] - 1) {
				// current hypothesis is at a word boundary -> expand it
				for (WordIdx word_idx = 0; word_idx < n_words; word_idx++) {

					double   word_penalty = word_idx != lexicon_.silence_idx() ? word_penalty_ : 0.0;
					StateIdx first_state  = lexicon_.get_automaton_for_word(word_idx).first_state();
					for (size_t init_state = 0; init_state <= 1; init_state++) {

						Book&    target_hyp  = next_hyps[max_states_per_word * word_idx + init_state];
						double   new_score   = cur_hyp->score + word_penalty + tdp_model_.score(first_state, init_state+1);

						// check if the hypothesis is already bad enough
						if (new_score > target_hyp.score) {
							continue;
						}

						// acoustic model scoring
						if (am_cache[first_state] == std::numeric_limits<double>::infinity()) {
							am_cache[first_state] = scorer_.score(cur_frame, first_state);
						}
						new_score += am_cache[first_state];

						if (target_hyp.score > new_score) {
							merge_hypothesis(new_score, init_state, t - 1, word_idx, target_hyp);
							best_score = std::min(best_score, new_score);
						}
					}
				}
			} else {

				// expand the 0-1-2 topology
				for (size_t jump = 0; jump <= 2; jump++) {
					StateIdx next_state_idx = cur_hyp->state_idx + jump;
					if ((size_t) next_state_idx >= n_states_per_word[cur_hyp->word]) {
						break;
					}

					StateIdx automaton_state =  lexicon_.get_automaton_for_word(cur_hyp->word)[next_state_idx];
					Book&    target_hyp = next_hyps[max_states_per_word * cur_hyp->word + next_state_idx];
					double   new_score  = cur_hyp->score + tdp_model_.score(automaton_state, jump);

					// check if the hypothesis is already bad enough
					if (new_score > target_hyp.score) {
						continue;
					}

					// acoustic model scoring
					if (am_cache[automaton_state] == std::numeric_limits<double>::infinity()) {
						am_cache[automaton_state] = scorer_.score(cur_frame, automaton_state);
					}
					new_score += am_cache[automaton_state];

					if (target_hyp.score > new_score) {
						merge_hypothesis(new_score, next_state_idx, cur_hyp->bkp, cur_hyp->word, target_hyp);
						best_score = std::min(best_score, new_score);
					}
				}
			}
		}

		traceback[t].score = std::numeric_limits<double>::infinity();
		for (std::vector<Book>::iterator cur_hyp = next_hyps.begin(); cur_hyp != next_hyps.end(); cur_hyp++) {

			// Prune the current hypothesis
			if (cur_hyp->score > best_score + am_threshold_) {
				cur_hyp->score = std::numeric_limits<double>::infinity();
				continue;
			}

			// Set the traceback arrays at word endings
			if ((size_t) cur_hyp->state_idx == lexicon_.get_automaton_for_word(cur_hyp->word).num_states() - 1) {
				if (traceback[t].score > cur_hyp->score) {
					traceback[t].score = cur_hyp->score;
					traceback[t].word  = cur_hyp->word;
					traceback[t].bkp 	 = cur_hyp->bkp;
				}
			}
		}
/*
		std::cout << "Traceback info: " << t << " "
																		<< traceback[t].score << " "
																		<< traceback[t].word  << " "
																		<< traceback[t].bkp   << " "
																		<< best_score << std::endl;
																		*/
		next_hyps.swap(current_hyps);
		//std::copy(next_hyps.begin(), next_hyps.end(), current_hyps.begin());
		std::fill(next_hyps.begin(), next_hyps.end(), Book(std::numeric_limits<double>::infinity(), 0, 0, 0));
		std::fill(am_cache.begin() , am_cache.end() , std::numeric_limits<double>::infinity());
	}

	output.clear();
	t = n_features;
	//std::cout << "Best score (pruned): " << traceback[n_features].score << std::endl;
	while (t > 0) {
		if (traceback[t].word != lexicon_.silence_idx()){
			output.push_back(traceback[t].word);
		}
		t = traceback[t].bkp;
	}
	std::reverse(output.begin(), output.end());
}

void Recognizer::recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {
  // TODO: implement
	size_t num_features = feature_end - feature_begin;
	size_t num_states 	= lexicon_.num_states();
	size_t num_words 		= lexicon_.num_words();
	// set output size to zero
	output.resize(0);

	scorer_.prepare_sequence(feature_begin, feature_end);

	// store number of states for each word + one extra state for word boundary
	size_t num_states_word[num_words];
	MarkovAutomaton current_automaton;
	for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
		current_automaton = lexicon_.get_automaton_for_word(word_idx);
		num_states_word[word_idx] = current_automaton.num_states() + 1;
	}
	// backtrace information about best ending word for each time frame
	Book book[num_features];
	// hypothesis
	Book hyp[num_words][num_states];
	Book hypTmp[num_states];

	// initialise hypothesis with infinity score
	for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) {
		for (StateIdx state_idx = 0; state_idx <= num_states; state_idx++) {
			hyp[word_idx][state_idx].score = std::numeric_limits<double>::infinity();
		}
	}

	// costs for virtual state
	book[0].score = 0.0;
	size_t frame_counter = 1;
	double tmp_score;
	double current_word_penalty;
	for (FeatureIter iter = feature_begin; iter != feature_end; iter++, frame_counter++) { // loop features

		for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
			//word transition: Q(t-1,0; w) = Q(t-1, S(W(t-1)); W(t-1)) - log p(w)
			current_word_penalty = word_idx != lexicon_.silence_idx() ? word_penalty_ : 0.0;
			hyp[word_idx][0].score = book[frame_counter-1].score + current_word_penalty; // -log p(w) = const -> zerogram
			hyp[word_idx][0].bkp 	 = frame_counter - 1;

			current_automaton = lexicon_.get_automaton_for_word(word_idx);
			for (StateIdx state_idx = 1; state_idx < num_states_word[word_idx]; state_idx++) { // loop states
				// set score for current state to infinity
				hypTmp[state_idx].score = std::numeric_limits<double>::infinity();
				// find best predecessor state
				for (StateIdx pre = std::max(0, state_idx - 2); pre <= state_idx; pre++) { // loop over predecessor states
					// compute tmp score + tdp
					tmp_score = hyp[word_idx][pre].score + tdp_model_.score(current_automaton[pre], state_idx - pre);
					// store state hypothesis of current word temporally
					if (tmp_score < hypTmp[state_idx].score) {
						hypTmp[state_idx].score = tmp_score;
						hypTmp[state_idx].bkp   = hyp[word_idx][pre].bkp;
					}
				}
			}
			// store hypothesis
			for (StateIdx state_idx = 1; state_idx < num_states_word[word_idx]; state_idx++) { // loop states
				hyp[word_idx][state_idx].bkp 	 = hypTmp[state_idx].bkp;
				hyp[word_idx][state_idx].score = hypTmp[state_idx].score  + scorer_.score(iter, current_automaton[state_idx-1]);
			}
		} // loop words

		// find best ending word and store its score and start time
		book[frame_counter].score = std::numeric_limits<double>::infinity();
		for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
			if (hyp[word_idx][num_states_word[word_idx]-1].score < book[frame_counter].score) {
				book[frame_counter].score = hyp[word_idx][num_states_word[word_idx]-1].score;
				book[frame_counter].bkp 	= hyp[word_idx][num_states_word[word_idx]-1].bkp;
				book[frame_counter].word  = word_idx;
			}
		}
/*
		std::cout << "Traceback info: " << frame_counter<< " "
																		<< book[frame_counter].score << " "
																		<< book[frame_counter].word << " "
																		<< book[frame_counter].bkp << std::endl;
																		*/
	} // loop features

	//std::cout << "Best score (unpruned): " << book[num_features].score << std::endl;
	// trace back
	size_t count = 0;
	size_t feature_index = num_features;
	while (feature_index > 0) {
		if (book[feature_index].word != lexicon_.silence_idx()){
			output.push_back(book[feature_index].word);
			count++;
		}
		feature_index = book[feature_index].bkp;
	}
	std::reverse(output.begin(), output.end());
}

/*****************************************************************************/

EDAccumulator Recognizer::editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end) {
  EDAccumulator result;
  size_t ref_size = ref_end - ref_begin;
  size_t hyp_size = rec_end - rec_begin;

  // Initialize the vectors containing the accumulators
  // with the dynamic programming initialization equations
  std::vector<EDAccumulator> current_rates(1+ref_size, EDAccumulator());
  for (size_t ref_idx = 1; ref_idx <= ref_size; ref_idx++) {
  	current_rates[ref_idx] = current_rates[ref_idx-1];
  	current_rates[ref_idx].deletion_error();
  }
  std::vector<EDAccumulator> previous_rates (1+ref_size, EDAccumulator());

  for (size_t hyp_idx = 1; hyp_idx <= hyp_size; hyp_idx++) {
  	// reset accumulators
  	current_rates.swap(previous_rates);

  	// add an insertion error for the very first accumulator (DP initialization)
  	current_rates[0].insertion_error();

  	// parse through all reference indices
  	for (size_t ref_idx = 1; ref_idx <= ref_size; ref_idx++) {
  		uint16_t best_count = 0xFFFF;

  		// Words are equal -> move diagonally (with no error)
  		if (previous_rates[ref_idx-1].total_count < best_count && *(ref_begin+ref_idx-1) == *(rec_begin+hyp_idx-1)) {
  			current_rates[ref_idx] = previous_rates[ref_idx-1];
  			best_count = current_rates[ref_idx].total_count;
  		}

  		// Move diagonally and account for a substitution error
  		if (previous_rates[ref_idx-1].total_count + 1 < best_count) {
  			current_rates[ref_idx] = previous_rates[ref_idx-1];
  			current_rates[ref_idx].substitution_error();
  			best_count = current_rates[ref_idx].total_count;
  		}

  		// Move vertically and account for an insertion error
  		if (previous_rates[ref_idx].total_count + 1 < best_count) {
  			current_rates[ref_idx] = previous_rates[ref_idx];
  			current_rates[ref_idx].insertion_error();
  			best_count = current_rates[ref_idx].total_count;
  		}

  		// Move horizontally and account for a deletion error
  		if (current_rates[ref_idx-1].total_count + 1 < best_count) {
  			current_rates[ref_idx] = current_rates[ref_idx-1];
  			current_rates[ref_idx].deletion_error();
  			best_count = current_rates[ref_idx].total_count;
  		}
  	}
  }

  // In this position we have calculated the optimal distance between the two word sequences
  result = current_rates[ref_size];
  return result;
}

/*****************************************************************************/
