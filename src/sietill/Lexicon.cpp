/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Lexicon.hpp"

#include <iostream>

WordIdx Lexicon::add_word(std::string const& orth, uint16_t num_states, uint16_t state_repetitions, bool silence) {
  WordIdx word_idx = automata_.size();
  if (silence) {
    silence_ = word_idx;
  }
  StateIdx start_state = automata_.empty() ? 0u : (automata_[automata_.size()-1u].last_state() + 1u);

  orth_.push_back(orth);
  automata_.push_back(MarkovAutomaton(start_state, num_states, state_repetitions));

  return word_idx;
}

/*****************************************************************************/

MarkovAutomaton const& Lexicon::get_silence_automaton() const {
  return automata_[silence_];
}

/*****************************************************************************/

MarkovAutomaton const& Lexicon::get_automaton_for_word(WordIdx word_idx) const {
  return automata_[word_idx];
}

/*****************************************************************************/

StateIdx Lexicon::num_states() const {
  return automata_[automata_.size() - 1u].last_state() + 1u;
}

/*****************************************************************************/

WordIdx Lexicon::num_words() const {
  return automata_.size();
}

/*****************************************************************************/

WordIdx Lexicon::silence_idx() const {
  return silence_;
}

/*****************************************************************************/

WordIdx Lexicon::operator[](std::string const& orth) const {
  // TODO: faster search
  auto iter = std::find(orth_.begin(), orth_.end(), orth);
  if (iter == orth_.end()) {
    std::cerr << "unknown word: '" << orth << "'" << std::endl; 
    return -1; // TODO: add unknown word
  }
  else {
    return iter - orth_.begin();
  }
}

/*****************************************************************************/

Lexicon build_sietill_lexicon() {
  Lexicon result;
  result.add_word("[silence]",  1, 1, true);
  result.add_word(     "eins", 18, 2);
  result.add_word(     "zwei", 18, 2);
  result.add_word(     "drei", 18, 2);
  result.add_word(     "vier", 18, 2);
  result.add_word(    "fuenf", 18, 2);
  result.add_word(    "sechs", 18, 2);
  result.add_word(   "sieben", 18, 2);
  result.add_word(     "acht", 18, 2);
  result.add_word(     "neun", 18, 2);
  result.add_word(     "null", 18, 2);
  result.add_word(      "zwo", 18, 2);
  return result;
}

/*****************************************************************************/
