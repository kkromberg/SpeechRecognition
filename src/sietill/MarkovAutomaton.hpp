/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __MARKOV_AUTOMATON_HPP__ 
#define __MARKOV_AUTOMATON_HPP__

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "Types.hpp"

struct MarkovAutomaton {
  std::vector<StateIdx> states;

  MarkovAutomaton() {}

  MarkovAutomaton(StateIdx start, uint16_t num, uint16_t repetitions) : states(num*repetitions) {
    std::vector<StateIdx>::iterator iter = states.begin();
    for (StateIdx s = start; s < start + num; s++) {
      std::fill_n(iter, repetitions, s);
      iter += repetitions;
    }
  }

  StateIdx first_state() const {
    return states[0u];
  }

  StateIdx last_state() const {
    return states[states.size() - 1u];
  }

  size_t num_states() const {
    return states.size();
  }

  StateIdx max_state() const {
    return std::accumulate(states.begin(), states.end(), std::numeric_limits<StateIdx>::min(), static_cast<StateIdx const&(*)(StateIdx const&, StateIdx const&)>(std::max<StateIdx>));
  }

  StateIdx& operator[](size_t idx) {
    return states[idx];
  }

  StateIdx operator[](size_t idx) const {
    return states[idx];
  }

  static MarkovAutomaton concat(std::vector<MarkovAutomaton const*> automata) {
  	MarkovAutomaton result;
    //TODO: implement
    // iterate through automata
    unsigned int state_counter = 0;
    for (std::vector<MarkovAutomaton>::size_type i = 0; i != automata.size(); i++) {
    	// iterate through all states of the current automaton and append them to the result
    	for (std::size_t j = 0; j < ((MarkovAutomaton)*automata[i]).num_states(); j++) {
    		// append each state
    		result.states[state_counter] = ((MarkovAutomaton)*automata[i])[j];
    		state_counter++;
    	}
    }
    return result;
  }
};

#endif /* __MARKOV_AUTOMATON_HPP__ */
