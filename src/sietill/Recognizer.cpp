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
  std::cerr << "Beginning search:" << std::endl;
  const WordIdx  invalid_word  = lexicon_.num_words();
  const StateIdx virtual_state = lexicon_.num_states();
  size_t n_frames = feature_end - feature_begin;

  // initialize traceback arrays with default values
  std::vector<WordIdx> traceback_word(n_frames, invalid_word);
  std::vector<size_t > traceback_time(n_frames, 0);

  // arrays to keep track of the hypotheses that are kept in each time step
  Beam next_hypotheses(lexicon_.num_states() * lexicon_.num_words(), nullptr);

  // initialize the DP search with an empty hypothesis
  std::vector<size_t> beam_boundaries(n_frames + 2, 0);
  beam_boundaries[1] = 1;
  Beam hypothesis_beams;
  hypothesis_beams.push_back(new Hypothesis());

  size_t t = 0;
  for (FeatureIter feature_iter = feature_begin; feature_iter != feature_end - 1; feature_iter++, t++) {
    BeamIterator beam_start = hypothesis_beams.begin() + beam_boundaries[t];
    BeamIterator beam_end   = hypothesis_beams.begin() + beam_boundaries[t+1];
    std::cerr << "t = " << t << " " << beam_end - beam_start << std::endl;

    bool did_word_expansions = false;
    for (BeamIterator hypothesis_it = beam_start; hypothesis_it != beam_end; hypothesis_it++) {
      Hypothesis *current_hyp   = *hypothesis_it;
      StateIdx    current_state = current_hyp->state_;
      WordIdx     current_word  = current_hyp->word_;

      if (current_state >= ((int) lexicon_.get_automaton_for_word(current_word).num_states()) - 1) {
        did_word_expansions = true;
        // Hypothesis is at a word boundary -> we can start a new word (or silence)
        for (WordIdx next_word = 0; next_word < lexicon_.num_words(); next_word++) {
          std::cerr << "Expanded with word: " << next_word << std::endl;
          double current_word_penalty = next_word != lexicon_.silence_idx() ? word_penalty_ : 0.0;
          if (next_hypotheses[next_word] == nullptr) {

            // insert a new hypothesis for the word
            Hypothesis *new_hypothesis = new Hypothesis();

            new_hypothesis->ancestor_  = current_hyp;
            new_hypothesis->score_    += current_word_penalty;
            new_hypothesis->word_      = next_word;
            new_hypothesis->state_     = virtual_state;
            new_hypothesis->new_word_  = true;

            beam_boundaries[t+1]++;
            next_hypotheses[next_word] = new_hypothesis;
          } else if (current_hyp->score_ + current_word_penalty < next_hypotheses[next_word]->score_) {
            // replace the previous hypothesis with a better one
            next_hypotheses[next_word]->ancestor_  = current_hyp->ancestor_;
            next_hypotheses[next_word]->score_    += current_hyp->score_ + word_penalty_;
          }
        }
      }
    }
    std::cerr << "did word expansions" << std::endl;

    // move the new word hypotheses into the current beam, if there were word expansions
    // note there will always be as many word expansions as words in the vocabulary
    if (did_word_expansions) {
      hypothesis_beams.resize(hypothesis_beams.size() + lexicon_.num_words(), nullptr);

      std::copy(next_hypotheses.begin(), next_hypotheses.begin() + lexicon_.num_words(),
                hypothesis_beams.begin() + hypothesis_beams.size() - lexicon_.num_words());

      did_word_expansions = false;
      std::fill(next_hypotheses.begin(), next_hypotheses.end(), nullptr);

      beam_start = hypothesis_beams.begin() + beam_boundaries[t];
      beam_end   = hypothesis_beams.begin() + beam_boundaries[t+1];
    }

    // Skip the empty hypothesis at first
    if (t == 0) {
      beam_start++;
    }

    // expand all hypotheses in the beam
    double best_score_in_hyp = 0.0;
    for (BeamIterator hypothesis_it = beam_start; hypothesis_it != beam_end; hypothesis_it++) {
      std::cerr << "Hypothesis nr. " << hypothesis_it - beam_start << " " << std::endl;
      Hypothesis *current_hyp   = *hypothesis_it;
      StateIdx    current_state = current_hyp->state_;
      WordIdx     current_word  = current_hyp->word_;
      StateIdx    max_state     = lexicon_.get_automaton_for_word(current_word).num_states() - 1;

      for (size_t jump = 0; jump <= std::min(2, max_state - current_state); jump++) {
        std::cerr << "jump : " << jump << std::endl;

        if (jump == 0 && current_state == virtual_state) {
          // A hypothesis, which has just had a word expansion cannot have a loop transition
          // TODO: A hypothesis which "forwarded" into a word expansion can also not have a skip transition!
          // How to account for this?
          continue;
        }

        if (lexicon_.get_automaton_for_word(current_word).num_states() >= current_state + jump) {
          break;
        }

        // This is the literal state of the markov automaton. Not the state position!
        StateIdx next_state = lexicon_.get_automaton_for_word(current_word)[current_state + jump];

        // Get the score of the transition
        // TODO: Cache the am scores
        double new_score = scorer_.score(feature_iter + 1, next_state) + tdp_model_.score(next_state, jump);

        Hypothesis *new_hyp = next_hypotheses[(current_state + jump) * lexicon_.num_states() + current_word];
        if (new_hyp == nullptr) {
          // Create a new hypothesis
          new_hyp = new Hypothesis(current_hyp, current_hyp->score_ + new_score,
                                   current_state + jump, current_word, false);

        } else if (current_hyp->score_ + new_score < new_hyp->score_) {
          // Replace the current hypothesis with a better one
          new_hyp->ancestor_ = current_hyp->ancestor_;
          new_hyp->score_    = current_hyp->score_ + new_score;
        }

        best_score_in_hyp = std::min(new_hyp->score_, best_score_in_hyp);
      }

    }

    // reverse enough space for all new hypothesis in the beam container
    size_t next_position = hypothesis_beams.size();
    hypothesis_beams.resize(hypothesis_beams.size() + lexicon_.num_states() * lexicon_.num_words(), nullptr);

    // Do threshold pruning
    for (BeamIterator it = next_hypotheses.begin(); it != next_hypotheses.end(); it++) {
      if (*it == nullptr) continue;

      Hypothesis *hyp = *it;
      if (hyp->score_ > best_score_in_hyp + am_threshold_) {
        // too bad score
        delete hyp;
        hyp = nullptr;
      } else {
        // Hypothesis survived pruning
        hypothesis_beams[next_position++] = hyp;
      }
    }

    // Shrink container to fit the new hypotheses
    hypothesis_beams.resize(hypothesis_beams.size() - (lexicon_.num_states() * lexicon_.num_words()) + next_position);

    // update beam boundaries
    beam_boundaries[t+2] = beam_boundaries[t+1] + next_position;
  }


  std::cerr << "Search complete. Backtracking commencing." << std::endl;

  BeamIterator beam_start = hypothesis_beams.begin() + beam_boundaries[n_frames + 1];
  BeamIterator beam_end   = hypothesis_beams.begin() + beam_boundaries[n_frames + 2];
  Hypothesis  *best_hyp(nullptr);
  for (BeamIterator it = beam_start; it != beam_end; it++) {
    if (best_hyp == nullptr || best_hyp->score_ > (*it)->score_) {
      best_hyp = *it;
    }
  }

  while (best_hyp->ancestor_ != nullptr) {
    if (best_hyp->new_word_ && best_hyp->word_ != lexicon_.silence_idx()) {
      output.push_back(best_hyp->word_);
    }
    best_hyp = best_hyp->ancestor_;
  }

  std::reverse(output.begin(), output.end());
  std::cerr << "Backtracking done." << std::endl;

}

/*****************************************************************************/

EDAccumulator Recognizer::editDistance(WordIter ref_begin, WordIter ref_end, WordIter rec_begin, WordIter rec_end) {
  EDAccumulator result;
  // TODO: implement
  return result;
}

/*****************************************************************************/
