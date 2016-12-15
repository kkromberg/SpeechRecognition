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

#include "Recognizer.hpp"
#include "TdpModel.hpp"
#include "Timer.hpp"


/*****************************************************************************/

namespace {
}

/*****************************************************************************/

const ParameterDouble Recognizer::paramAmThreshold    ("am-threshold" ,   20.0);
const ParameterDouble Recognizer::paramWordPenalty    ("word-penalty" ,   10.0);
const ParameterBool   Recognizer::paramPrunedSearch   ("pruned-search",   true);


/*****************************************************************************/

void Recognizer::recognize(Corpus const& corpus) {
	Timer search_timer;
  EDAccumulator acc;
  size_t ref_total = 0ul;
  size_t sentence_errors = 0ul;
  std::vector<WordIdx> recognized_words;

  for (SegmentIdx s = 0ul; s < corpus.get_corpus_size(); s++) {
    std::pair<FeatureIter, FeatureIter> features = corpus.get_feature_sequence(s);
    std::pair<WordIter, WordIter> ref_seq = corpus.get_word_sequence(s);

    search_timer.tick();
    if (pruned_search_) {
      recognizeSequence_pruned(features.first, features.second, recognized_words);
    } else {
      recognizeSequence(features.first, features.second, recognized_words);
    }
    search_timer.tock();

    EDAccumulator ed = editDistance(ref_seq.first, ref_seq.second, recognized_words.begin(), recognized_words.end());
    acc += ed;
    ref_total += std::distance(ref_seq.first, ref_seq.second);
    if (ed.total_count > 0u) {
      sentence_errors++;
    }

    const double wer = (static_cast<double>(ed.total_count) / static_cast<double>(ref_seq.second - ref_seq.first)) * 100.0;
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
  const double time = search_timer.secs(); // TODO: compute
  const double rtf = 0.0; // TODO: compute

  std::cerr << "WER: " << std::setw(6) << std::fixed << wer << std::setw(0)
            << "% (S/I/D) " << acc.substitute_count << "/" << acc.insert_count << "/" << acc.delete_count << std::endl;
  std::cerr << "SER: " << std::setw(6) << std::fixed << ser << "%" << std::endl;
  std::cerr << "Time: " << time << " seconds" << std::endl;
  std::cerr << "RTF: " << rtf << std::endl;
}

/*****************************************************************************/

void Recognizer::recognizeSequence_pruned(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {

	// Variables to keep track of pruning information
	size_t hypotheses_pruned     = 0;
	size_t hypothesis_expansions = 0;
  size_t n_frames = feature_end - feature_begin;

  // Build an invalid hypothesis
  const StateIdx      virtual_state      = lexicon_.num_states();
  const HypothesisPtr invalid_hypothesis = HypothesisPtr(new Hypothesis(HypothesisPtr(nullptr),
        std::numeric_limits<double>::infinity(), virtual_state, lexicon_.silence_idx(), false));

  // arrays to keep track of the hypotheses that are kept in each time step
  Beam hyp_expansions(lexicon_.num_states() * lexicon_.num_words(), invalid_hypothesis);

  // initialize the DP search with an empty hypothesis
  std::vector<size_t> beam_boundaries(n_frames + 2, 0);
  beam_boundaries[1] = 1;
  Beam hypothesis_beams(1, HypothesisPtr(new Hypothesis()));

  std::vector<double> am_cache(lexicon_.num_states(), std::numeric_limits<double>::infinity());

  size_t t = 0;
  for (FeatureIter feature_iter = feature_begin; feature_iter != feature_end - 1; feature_iter++, t++) {
    BeamIterator beam_start = hypothesis_beams.begin() + beam_boundaries[t];
    BeamIterator beam_end   = hypothesis_beams.begin() + beam_boundaries[t+1];

    // Attempt to expand all hypotheses in word boundaries to a new word
    bool did_word_expansions = false;
    for (BeamIterator hypothesis_it = beam_start; hypothesis_it != beam_end; hypothesis_it++) {
      HypothesisPtr current_hyp   = *hypothesis_it;
      StateIdx      current_state = current_hyp->state_;
      WordIdx       current_word  = current_hyp->word_;
      double        current_score = current_hyp->score_;
      StateIdx      max_state     = lexicon_.get_automaton_for_word(current_word).num_states() - 1;

      if (current_hyp->is_initial() || max_state == current_state) {
        did_word_expansions = true;

        // Hypothesis is at a word boundary -> we can start a new word (or silence)
        for (WordIdx next_word = 0; next_word < lexicon_.num_words(); next_word++) {

        	// silence has a cost of 0, i.e. prob of 1
          double current_word_penalty = next_word != lexicon_.silence_idx() ? word_penalty_ : 0.0;

          if (hyp_expansions[next_word] == invalid_hypothesis) {
            // insert a new hypothesis for the word
          	// TODO: Avoid creating objects in the middle of the search algorithm
            hyp_expansions[next_word] = HypothesisPtr(new Hypothesis(current_hyp,
                                                         current_score + current_word_penalty,
                                                         virtual_state, next_word, true));

            // update the beam boundaries to account for the new hypothesis
            beam_boundaries[t+1]++;
          } else if (current_score + current_word_penalty < hyp_expansions[next_word]->score_) {
            // replace the previous hypothesis with a better one
            hyp_expansions[next_word]->ancestor_  = current_hyp->ancestor_;
            hyp_expansions[next_word]->score_     = current_score + current_word_penalty;
          }
        }
      }
    }

    // move the new word hypotheses into the current beam, if there were word expansions
    // note there will always be as many word expansions as words in the vocabulary
    if (did_word_expansions) {
      did_word_expansions = false;

      // copy all word expansions into the beam
      hypothesis_beams.resize(hypothesis_beams.size() + lexicon_.num_words(), invalid_hypothesis);

      std::copy(hyp_expansions.begin(), hyp_expansions.begin() + lexicon_.num_words(),
                hypothesis_beams.begin() + hypothesis_beams.size() - lexicon_.num_words());

      // reset the expansion container
      std::fill(hyp_expansions.begin(), hyp_expansions.end(), invalid_hypothesis);

      // update iterators, since the resizing operation has shifted the boundaries
      beam_start = hypothesis_beams.begin() + beam_boundaries[t];
      beam_end   = hypothesis_beams.begin() + beam_boundaries[t+1];
    }

    // Skip the empty hypothesis at first
    if (t == 0) {
      beam_start++;
    }

    // keep track of the best score in the beam for pruning afterwards
    double best_score_in_hyp = std::numeric_limits<double>::infinity();
    std::fill(am_cache.begin(), am_cache.end(), std::numeric_limits<double>::infinity());

    // expand all hypotheses in the beam
    for (BeamIterator hypothesis_it = beam_start; hypothesis_it != beam_end; hypothesis_it++) {
      HypothesisPtr current_hyp   = *hypothesis_it;
      StateIdx      current_state = current_hyp->state_;
      WordIdx       current_word  = current_hyp->word_;
      StateIdx      max_state     = lexicon_.get_automaton_for_word(current_word).num_states() - 1;
      double        current_score = current_hyp->score_;

      // Handle the case when the previous hypothesis was at a word boundary
      // In this case, we cannot have a forward transition
      bool ignore_forward_jump = current_state == virtual_state ? true : false;
      current_state = current_state == virtual_state ? 0 : current_state;

      for (int jump = 0; jump <= std::min(2, max_state - current_state); jump++) {

        if (max_state < current_state + jump || (ignore_forward_jump && jump == 2)) {
        	// The current and all subsequent jumps are forbidden. Break the loop
          break;
        }

        // This is the literal state of the markov automaton. Not the state position!
        StateIdx next_state = lexicon_.get_automaton_for_word(current_word)[current_state + jump];

        // Get the cached score of the transition
        if (am_cache[next_state] == std::numeric_limits<double>::infinity()) {
        	// TODO: Account for context frames when using neural models
        	scorer_.prepare_sequence(feature_iter + 1, feature_iter + 2);

        	am_cache[next_state] = scorer_.score(feature_iter + 1, next_state);
        }
        double new_score = am_cache[next_state] + tdp_model_.score(next_state, jump);

        size_t container_index = (current_state + jump) * lexicon_.num_words() + current_word;
        HypothesisPtr new_hyp = hyp_expansions[container_index];

        if (new_hyp == invalid_hypothesis || current_score + new_score < new_hyp->score_) {
          // Create a new hypothesis or replace the current hypothesis with a better one
        	// TODO: Avoid creating objects in the middle of the search algorithm
          hyp_expansions[container_index] = HypothesisPtr(new Hypothesis(current_hyp,
          		                                               current_score + new_score,
                                                             current_state + jump, current_word, false));
        }

        // update the best score in the hypothesis
        best_score_in_hyp = std::min(new_hyp->score_, best_score_in_hyp);
      } // for(int jump = 0; ...)
    } // for(BeamIterator hypothesis_it = ...)

    // reserve enough space for all new hypothesis in the beam container
    size_t old_beam_size = hypothesis_beams.size();
    hypothesis_beams.resize(old_beam_size + lexicon_.num_states() * lexicon_.num_words(), invalid_hypothesis);

    // Do threshold pruning
    size_t next_hyp_idx = 0;
    BeamIterator next_hypotheses_begin = hyp_expansions.begin();
    BeamIterator next_hypotheses_end   = hyp_expansions.end();
    for (BeamIterator it = next_hypotheses_begin; it != next_hypotheses_end; it++) {
    	HypothesisPtr hyp = *it;

      if (hyp == invalid_hypothesis || hyp->new_word_ == true) {
      	continue;
      }

      /*
    	std::cout << std::setprecision(6) << "Q / w / s / bp / n: " << hyp->score_ << " "
    																			<< hyp->word_ << " "
    																			<< hyp->state_ << " "
    			                                << hyp->ancestor_ << " "
    			                                << hyp << " "
    			                                << (!hyp->new_word_ ? "false" : "true");
	*/
      if (hyp->score_ > best_score_in_hyp + am_threshold_) {
      	hypotheses_pruned++;
      	//std::cout << " Pruned" << std::endl;
        // too bad score
      } else {
      	//std::cout << " Kept" << std::endl;
      	hypothesis_expansions++;
        // Hypothesis survived pruning
        hypothesis_beams[old_beam_size + next_hyp_idx] = hyp;
        next_hyp_idx++;
      }
    } // for(BeamIterator it...)

    // Shrink container to fit the new hypotheses
    hypothesis_beams.resize(old_beam_size + next_hyp_idx);
    std::fill(hyp_expansions.begin(), hyp_expansions.end(), invalid_hypothesis);

    // update beam boundaries
    beam_boundaries[t+2] = hypothesis_beams.size();

    /*
    std::cout << "Pruning information: " << hypotheses_pruned     << " pruned. "
    		                                 << hypothesis_expansions << " kept at time step " << t << " "
    		                                 << "Best score: " << best_score_in_hyp << std::endl;
	*/
    hypotheses_pruned = hypothesis_expansions = 0;
  }

  // Find the best hypothesis in the final beam
  BeamIterator  beam_start = hypothesis_beams.begin() + beam_boundaries[n_frames - 1];
  BeamIterator  beam_end   = hypothesis_beams.begin() + beam_boundaries[n_frames ];
  HypothesisPtr best_hyp(*beam_start);
  for (BeamIterator it = beam_start; it != beam_end; it++) {
  	if (best_hyp->score_ > (*it)->score_) {
      best_hyp = *it;
    }
  }

  // Perform back tracking to extract the word sequence
  output.clear();
  while (1) {
    if (best_hyp->new_word_ && best_hyp->word_ != lexicon_.silence_idx()) {
    	// We found a hypothesis in a word boundary
      output.push_back(best_hyp->word_);
    }
    best_hyp = best_hyp->ancestor_;

    if (best_hyp->is_initial()) break;
  }

  std::reverse(output.begin(), output.end());
}
void Recognizer::recognizeSequence(FeatureIter feature_begin, FeatureIter feature_end, std::vector<WordIdx>& output) {
  // TODO: implement
	size_t num_features = feature_end - feature_begin;
	size_t num_states 	= lexicon_.num_states();
	size_t num_words 		= lexicon_.num_words();
	size_t num_states_word[num_words];	// contains number of states for each word
	output.resize(0);
	// save number of states for each word
	MarkovAutomaton current_automaton;
	for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
		current_automaton = lexicon_.get_automaton_for_word(word_idx);
		num_states_word[word_idx] = current_automaton.num_states() + 1; // one extra state for word boundary
	}
	/*
	std::cerr << "NUM FEATURES: "    << num_features    << std::endl;
	std::cerr << "NUM STATES: "      << num_states      << std::endl;
	std::cerr << "NUM WORDS: "       << num_words       << std::endl;
	for (size_t i = 0; i < sizeof(num_states_word)/sizeof(*num_states_word); i++ ) {
		std::cerr << "WORD: " << i << std::endl;
		std::cerr << "NUM STATES/WORD: " << num_states_word[i] << std::endl;
	}
	*/
	Book book[num_features];
	Book hyp[num_words][num_states];
	Book hypTmp[num_states];

	// initialize hypothesis with infinity score
	for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) {
		for (StateIdx state_idx = 0; state_idx <= num_states; state_idx++) {
			hyp[word_idx][state_idx].score = std::numeric_limits<double>::infinity();
		}
	}

	// costs for virtual state
	book[0].score = 0.0;
	size_t frame_counter = 1;
	double word_penalty =0;
	double tmp_score;
	for (FeatureIter iter = feature_begin; iter != feature_end; iter++, frame_counter++) { // loop features

		//std::cerr << "FRAME: " << frame_counter << std::endl;
		for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
			//std::cerr << "WORD: " << word_idx << std::endl;
			// word transition
			// Q(t-1,0; w) = Q(t-1, S(W(t-1)); W(t-1)) - log p(w)
			//std::cerr << book[0].score << std::endl;
			double current_word_penalty = word_idx != lexicon_.silence_idx() ? word_penalty_ : 0.0;
			hyp[word_idx][0].score = book[frame_counter-1].score + current_word_penalty; // -log p(w) = const -> zerogram
			//std::cerr << "SCORE FOR BETWEEN WORD INIT: " << hyp[word_idx][0].score << std::endl;
			//std::cin >> b;
			hyp[word_idx][0].bkp 	 = frame_counter - 1;
			// B(t-1,0; w) = t-1

			current_automaton = lexicon_.get_automaton_for_word(word_idx);

			for (StateIdx state_idx = 1; state_idx < num_states_word[word_idx]; state_idx++) { // loop states
				// set score for current state to infinity
				//std::cerr << "CURRENT STATE: " << state_idx << std::endl;
				hypTmp[state_idx].score = std::numeric_limits<double>::infinity();
				for (StateIdx pre = std::max(0, state_idx - 2); pre <= state_idx; pre++) { // loop over predecessor states
					// compute tmp score + tdp
					tmp_score = hyp[word_idx][pre].score + tdp_model_.score(current_automaton[pre], state_idx - pre);

					/*
					std::cerr << "PREVOIS STATE: " << pre << std::endl;
					std::cerr << "JUMP: " << state_idx - pre << std::endl;
					std::cerr << "TDP SCORE: " << tdp_model_.score(current_automaton[pre], state_idx - pre) << std::endl;
					std::cerr << "TMP SCORE: " << tmp_score << std::endl;
					std::cin >> b;
					*/
					//std::cerr << "HYP SCORE: " << hypTmp[state_idx].score << std::endl;
					if (tmp_score < hypTmp[state_idx].score) {
						hypTmp[state_idx].score = tmp_score;
						hypTmp[state_idx].bkp   = hyp[word_idx][pre].bkp;
					}

				}
				//std::cerr << "TMP SCORE: " << hypTmp[state_idx].score << std::endl;
				//std::cin >> b;
			}
				// store hypothesis
			for (StateIdx state_idx = 1; state_idx < num_states_word[word_idx]; state_idx++) { // loop states
				hyp[word_idx][state_idx].bkp 	 = hypTmp[state_idx].bkp;
				//Scorer is broken!!!
				hyp[word_idx][state_idx].score = hypTmp[state_idx].score  + scorer_.score(iter, current_automaton[state_idx-1]);
				//std::cerr << "SCORER: " << scorer_.score(iter, current_automaton[state_idx-1]) << std::endl;
			}
			 // loop states
		} // loop words
		book[frame_counter].score = std::numeric_limits<double>::infinity();
		// store best score, state and word into book
		for (WordIdx word_idx = 0; word_idx < num_words; word_idx++) { // loop words
			//std::cerr << "WORD IDX: " << word_idx << std::endl;
			//std::cerr << "SCORE: " << hyp[word_idx][num_states_word[word_idx]-1].score << std::endl;
			//std::cerr << "I AM HERE" << std::endl;
			//std::cerr << "HYPOTHESIS SCORE: " << hyp[word_idx][num_states_word[word_idx]].score << std::endl;
			//std::cerr << "BOOK SCORE: " << book[frame_counter].score << std::endl;
			if (hyp[word_idx][num_states_word[word_idx]-1].score < book[frame_counter].score) {

				//std::cerr << "WORD IDX: " << word_idx << std::endl;
				book[frame_counter].score = hyp[word_idx][num_states_word[word_idx]-1].score;
				book[frame_counter].bkp 	= hyp[word_idx][num_states_word[word_idx]-1].bkp;
				//std::cerr << "Current word idx: " << word_idx << std::endl;
				book[frame_counter].word  = word_idx;
				//std::cerr << "Writing word index into book: " << word_idx << std::endl;

			}
		}
		//std::cerr << "WRITING WORD IDX: " << book[frame_counter].word << std::endl;
		//std::cin >> b;
	} // loop features

	// trace back
	size_t count = 0;
	size_t feature_index = num_features - 1;

	while (feature_index > 0) {
		//std::cerr << "WRITING WORD IDX: " << book[feature_index].word << std::endl;
			//output[count] = book[feature_index].word;
		if (book[feature_index].word != lexicon_.silence_idx()){
			output.push_back(book[feature_index].word);
			count++;
		}
		//std::cerr << "NEW FEATURE IDX: " << book[feature_index].bkp << std::endl;
		feature_index = book[feature_index].bkp;

	}
	std::reverse(output.begin(), output.end());
	//std::cerr << "NUMBER OF WORDS: " << count << std::endl;
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
