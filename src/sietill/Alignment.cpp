/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Alignment.hpp"
#include "Util.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <limits>

/*****************************************************************************/

namespace {

	template<typename T>
	struct dump_vector {
		dump_vector(std::ostream& stream, std::vector<T> const& vector) {
			for (auto it : vector ) {
				stream << it << " ";
			}
			stream << "\n";
		}
	};
  // compute -log(exp(-a) + exp(-b)) using the following equality:
  // -log(exp(-a) + exp(-b)) = -log(exp(-a-c) + exp(-b-c)) - c
  //                         = -log(1 + exp(-abs(a-b))) - c
  // where c = max(-a, -b) = -min(a, b)
  double logsum(double a, double b) {
    const double diff = a - b;
    // if a = b = inf then a - b = nan, but for logsum the result is well-defined (inf)
    if (diff != diff) {
      return std::numeric_limits<double>::infinity();
    }
    return -log1p(std::exp(-std::abs(diff))) + std::min(a, b);
  }
}

/*****************************************************************************/

Aligner::Aligner(MixtureModel const& mixtures, TdpModel const& tdp_model, size_t max_aligns)
                 : mixtures_(mixtures), tdp_model_(tdp_model), max_aligns_(max_aligns) {}

/*****************************************************************************/

double Aligner::align_sequence_full(FeatureIter feature_begin, FeatureIter feature_end,
                                    MarkovAutomaton const& reference,
                                    AlignmentIter align_begin, AlignmentIter align_end) {

	BackpointerMatrix backpointer_matrix;
	int feature_number = feature_end - feature_begin;
	int state_number   = reference.num_states();
	// initialise cost and backpointer matrix
	for (StateIdx state = 0; state < state_number; state++){
		backpointer_matrix.push_back(std::vector<int>(feature_number, -1));													  	 // backpointer initially with zeros
	}

	std::vector<double> previous_costs = std::vector<double>(feature_number, std::numeric_limits<double>::infinity());
	std::vector<double> current_costs  = std::vector<double>(feature_number, std::numeric_limits<double>::infinity());

	// cost variables
	double local_costs   = 0.0,
				 loop_costs    = std::numeric_limits<double>::infinity(),
				 forward_costs = std::numeric_limits<double>::infinity(),
				 skip_costs    = std::numeric_limits<double>::infinity();

	// the 0-1-2 topology makes it so that certain are unreachable
	// these are the initial values for these states
	int max_state = 2;
	int min_state = state_number-1-(feature_number-2)*2;

	// costs for the first point are fixed
	previous_costs[0] = mixtures_.score(feature_begin, reference[0]);

	size_t feature_index = 1;
	for (FeatureIter feature_iter = feature_begin+1;
			 feature_iter != feature_end;
			 feature_iter++, feature_index++, min_state += 2, max_state += 2) { // loop features

		// Keep track of the previous state to avoid multiple computations of the emission probability
		StateIdx previous_state = -1;

		for (StateIdx state = std::max(0, min_state); state <= std::min(state_number-1, max_state); state++) { // loop states
			// compute local costs / emission probability if necessary
			if (previous_state != state){
				local_costs = mixtures_.score(feature_iter, reference[state]);
			}

			// compute transition plus corresponding penalty costs
			// (s)(t-1) loop costs: previous costs + tdp
			loop_costs = previous_costs[state] + tdp_model_.score(reference[state], 0);
			double best_costs = loop_costs;
			int taken_transition = 0;
			if (state > 0) {
				// (s-1)(t-1): forward case
				forward_costs = previous_costs[state-1] + tdp_model_.score(reference[state-1], 1);
				if (forward_costs < best_costs){
					best_costs = forward_costs;
					taken_transition = 1;
				}
			}
			if (state > 1) {
				// (s-2)(t-1): skip case
				skip_costs = previous_costs[state-2] + tdp_model_.score(reference[state-2], 2);
				if (skip_costs < best_costs) {
					best_costs = skip_costs;
					taken_transition = 2;
				}
			}
			// store minimal costs for current point
			current_costs[state] = local_costs + best_costs;
			// determine where the best transition came from
			backpointer_matrix[state][feature_index] = state - taken_transition;

			// update the previous state to avoid multiple computations of the emission probability
			previous_state = state;
		}

		// turn the current cost array into the previous cost array
		std::copy(current_costs.begin(), current_costs.end(), previous_costs.begin());
	}


	//std::cout << "New alignment: " << std::endl;
	feature_index = feature_number - 1;
	size_t state_index   = state_number - 1;
	for (AlignmentIter align_iter = align_end-1; align_iter != align_begin - 1; align_iter--, feature_index--) { // loop alignment
		StateIdx automaton_state = reference[state_index];
		(*align_iter)->state = automaton_state;

		//std::cout << feature_index << " " << state_index << std::endl;
		state_index = backpointer_matrix[state_index][feature_index];
	}

	// return the path costs
  return current_costs[state_number-1];
}


/*****************************************************************************/

double Aligner::align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                                      MarkovAutomaton const& reference,
                                      AlignmentIter align_begin, AlignmentIter align_end,
                                      double pruning_threshold) {
  // TODO: implement
	CostMatrix cost_matrix;
	BackpointerMatrix backpointer_matrix;
	int feature_number = feature_end - feature_begin;
	int state_number   = reference.num_states();
	// initialise cost and backpointer matrix
	for (StateIdx state = 0; state < state_number; state++){
		cost_matrix.push_back(std::vector<double>(feature_number, std::numeric_limits<double>::infinity())); // set costs to infinity
		backpointer_matrix.push_back(std::vector<int>(feature_number, 0));													  	 // backpointer initially with zeros
	}
	double local_costs = std::numeric_limits<double>::infinity();
	size_t max_state = 2;
	size_t min_state = state_number-1-(feature_number-2)*2;
	size_t t = 1;

	// costs for the first point are fixed
	cost_matrix[0][0] = mixtures_.score(feature_begin, reference[0]);

	// iterate through features
	for (FeatureIter feature_iter = feature_begin+1; feature_iter != feature_end; feature_iter++,t++,
	min_state += 2, max_state += 2) {
		int min = min_state;
		int max = max_state;
		double min_cost = std::numeric_limits<double>::infinity();

		//for the repetitions
		StateIdx previous_state=-1;

		//iterate through states
		for (StateIdx state = std::max(0, min); state <= std::min(state_number-1, max); state++) {
			if(previous_state!=state){
				local_costs = mixtures_.score(feature_iter, reference[state]);
			}
			double best = std::numeric_limits<double>::infinity();
			StateIdx state_prime=0;
			for (size_t i=std::max(0,state-max+2); i<=std::min(state+0,2);i++){
				double temp=cost_matrix[state-i][t-1] + tdp_model_.score(reference[state-i], i);
				if(temp<best){
					best=temp;
					state_prime=state-i;
				}
			}
			cost_matrix[state][t] = local_costs + best;

			//find the best hypothesis
			if (cost_matrix[state][t]<min_cost){
				min_cost = cost_matrix[state][t];
			}
			backpointer_matrix[state][t]=state_prime;
			previous_state=state;
		}

		//traverse through the states to discard bad hypothesizes
		for (StateIdx state = std::max(0, min); state < std::min(state_number-1, max); state++) {
			if (cost_matrix[state][t]>min_cost+pruning_threshold) {
				cost_matrix[state][t] = std::numeric_limits<double>::infinity();
			}
		}
	}
	// mapping of automaton states to alignment
	size_t feature_index = feature_number - 1;
	size_t state_index   = state_number - 1;
	for (AlignmentIter align_iter = align_end-1; align_iter != align_begin - 1; align_iter--, feature_index--) { // loop
		StateIdx automaton_state = reference[state_index];
		(*align_iter)->state = automaton_state;
		state_index = backpointer_matrix[state_index][feature_index];
	}
	// return the cost of best path
	return cost_matrix[state_number-1][feature_number-1];
}

/*****************************************************************************/

void dump_alignment(std::ostream& out, Alignment const& alignment, size_t max_aligns) {
  for (size_t f = 0ul; f < (alignment.size() / max_aligns); f++) {
    for (size_t a = 0ul; a < alignment[f * max_aligns].count; a++) {
      const size_t idx = f * max_aligns + a;
      out << f << " " << alignment[idx].state << " " << alignment[idx].weight << std::endl;
    }
  }
}

/*****************************************************************************/

void write_alignment(std::ostream& out, Alignment const& alignment, size_t max_aligns) {
  out.write(reinterpret_cast<char const*>(&max_aligns), sizeof(max_aligns));
  size_t num_frames = alignment.size() / max_aligns;
  out.write(reinterpret_cast<char const*>(&num_frames), sizeof(num_frames));
  out.write(reinterpret_cast<char const*>(alignment.data()), max_aligns * num_frames * sizeof(Alignment::value_type));
}

/*****************************************************************************/

void read_alignment(std::istream& in, Alignment& alignment, size_t& max_aligns) {
  in.read(reinterpret_cast<char*>(&max_aligns), sizeof(max_aligns));
  size_t num_frames = 0ul;
  in.read(reinterpret_cast<char*>(&num_frames), sizeof(num_frames));
  alignment.resize(num_frames * max_aligns);
  in.read(reinterpret_cast<char*>(alignment.data()), max_aligns * num_frames * sizeof(Alignment::value_type));
}

/*****************************************************************************/
