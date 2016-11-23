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


  // TODO: implement
	CostMatrix cost_matrix;
	BackpointerMatrix backpointer_matrix;
	int feature_number = feature_end - feature_begin;
	int state_number   = reference.num_states();
	// initialise cost and backpointer matrix
	for (StateIdx state = 0; state < state_number; state++){
		cost_matrix.push_back(std::vector<double>(feature_number, std::numeric_limits<double>::infinity())); // set costs to infinity
		backpointer_matrix.push_back(std::vector<int>(feature_number, -1));													  	 // backpointer initially with zeros
	}

	// cost variables
	double local_costs, loop_costs = std::numeric_limits<double>::infinity(),
				 forward_costs = std::numeric_limits<double>::infinity(),
				 skip_costs = std::numeric_limits<double>::infinity();

	int max_state = 2;
	int min_state = state_number-1-(feature_number-2)*2;
	// costs for the first point are fixed
	cost_matrix[0][0] = mixtures_.score(feature_begin, reference[0]);
	size_t t = 1;
	for (FeatureIter feature_iter = feature_begin+1; feature_iter != feature_end+1; feature_iter++,t++, min_state += 2, max_state += 2) { // loop features
		size_t best_state = 0;
		for (StateIdx state = std::max(0, min_state); state <= std::min(state_number-1, max_state); state++) { // loop states
			// compute local costs / emission probability
			local_costs = mixtures_.score(feature_iter, reference[state]);
			// compute transition plus corresponding penalty costs
			// (t-1) loop costs: previous costs + tdp
			loop_costs = cost_matrix[state][t-1] + tdp_model_.score(reference[state], 0);
			double best_costs = loop_costs;
			int taken_transition = 0;
			if (state > 0) {
				// (s-1)(n-1)
				forward_costs = cost_matrix[state-1][t-1] + tdp_model_.score(reference[state-1], 1);
				if (forward_costs < best_costs){
					best_costs = forward_costs;
					taken_transition = 1;
				}
			}
			if (state > 1) {
				// (s-2)(n-1)
				skip_costs = cost_matrix[state-2][t-1] + tdp_model_.score(reference[state-2], 2);
				if (skip_costs < best_costs) {
					best_costs = skip_costs;
					taken_transition = 2;
				}
			}
			// store minimal costs for current point
			cost_matrix[state][t] = local_costs + best_costs;

			// determine where the best transition came from
			backpointer_matrix[state][t] = state - taken_transition;
		}
	}


	//std::cout << "New alignment: " << std::endl;
	// TODO mapping of automaton states to alignment
	size_t feature_index = feature_number - 1;
	size_t state_index   = state_number - 1;
	for (AlignmentIter align_iter = align_end-1; align_iter != align_begin - 1; align_iter--, feature_index--) { // loop alignment
		StateIdx automaton_state = reference[state_index];
		(*align_iter)->state = automaton_state;

		//std::cout << feature_index << " " << state_index << std::endl;
		state_index = backpointer_matrix[state_index][feature_index];
	}

	// return the last entry from the cost matrix
  return cost_matrix[state_number-1][feature_number-1];
}

/*****************************************************************************/

double Aligner::align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                                      MarkovAutomaton const& reference,
                                      AlignmentIter align_begin, AlignmentIter align_end,
                                      double pruning_threshold) {
  // TODO: implement
  return 0.0;
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
