/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __LEXICON_HPP__
#define __LEXICON_HPP__

#include <string>
#include <vector>

#include "MarkovAutomaton.hpp"
#include "Types.hpp"

class Lexicon {
public:

  WordIdx add_word(std::string const& orth, uint16_t num_states, uint16_t state_repetitions, bool silence=false);

  MarkovAutomaton const& get_silence_automaton() const;
  MarkovAutomaton const& get_automaton_for_word(WordIdx word_idx) const;

  StateIdx num_states() const;
  WordIdx  num_words() const;
  WordIdx  silence_idx() const;

  WordIdx operator[](std::string const& orth) const;
private:
  std::vector<std::string>     orth_;
  std::vector<MarkovAutomaton> automata_;
  WordIdx                      silence_;
};

Lexicon build_sietill_lexicon();

#endif /* __LEXICON_HPP__ */
