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
    (*align_iter)[0].state  = automaton_state;
    (*align_iter)[0].weight = 1;
    (*align_iter)[0].count  = 1;
		//std::cout << feature_index << " " << state_index << std::endl;
		state_index = backpointer_matrix[state_index][feature_index];
	}

	//std::cout << "best costs: " << current_costs[state_number-1] << std::endl;

	// return the path costs
  return current_costs[state_number-1];
}


/*****************************************************************************/

double Aligner::align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                                      MarkovAutomaton const& reference,
                                      AlignmentIter align_begin, AlignmentIter align_end,
                                      double pruning_threshold) {
	size_t histogram_size = pruning_threshold;
	size_t n_features = feature_end - feature_begin;
	std::vector<double> emission_score_cache = std::vector<double>(reference.num_states(),
																																 std::numeric_limits<double>::infinity());

	std::vector<Beam> beams = std::vector<Beam>(n_features, Beam());
	Node *initial_node = new Node(NULL, 0, mixtures_.score(feature_begin, reference.first_state()));

	beams[0].insert(std::make_pair(0, initial_node));

	FeatureIter feature_iter = feature_begin + 1;
	for (size_t beam_index = 1; beam_index < n_features; beam_index++, feature_iter++) {

		// reset the emission score cache
		std::fill(emission_score_cache.begin(),
							emission_score_cache.end(),
							std::numeric_limits<double>::infinity());

		// loop over all previous hypotheses in beam of the previous feature vector
		for (BeamIterator previous_hyp_entry = beams[beam_index-1].begin();
				previous_hyp_entry != beams[beam_index-1].end();
				previous_hyp_entry++) {

			StateIdx  previous_hyp_state = previous_hyp_entry->first;
			Node     *previous_hyp_node  = previous_hyp_entry->second;

			// hypothesize the state changes of the 0-1-2 topology
			for (size_t jump = 0; jump <= 2; jump++) {
				StateIdx new_state_index = previous_hyp_state + jump;

				// Check if the node is still within our search space
				if (new_state_index >= reference.num_states()) {
					break;
				}

				// compute the costs of the new node
				double new_costs = previous_hyp_node->score;
				new_costs += tdp_model_.score(reference[new_state_index], jump);

				// check the cache if the value has already been computed
				if (emission_score_cache[new_state_index] == std::numeric_limits<double>::infinity()) {
					// compute the score
					emission_score_cache[new_state_index] = mixtures_ .score(feature_iter, reference[new_state_index]);
				}
				new_costs += emission_score_cache[new_state_index];

				// create a new node and add it to the beam
				Node *new_node = new Node(previous_hyp_node, new_state_index, new_costs);

				// try to find the node in the current beam (search by state index)
				BeamIterator same_state_node = beams[beam_index].find(new_state_index);

				if (same_state_node == beams[beam_index].end()) {
					// the element does not yet exist
					beams[beam_index].insert(std::make_pair(new_state_index, new_node));
				} else if (new_costs < same_state_node->second->score) {
					// the element exists, but has a worse score -> free the memory and set it
					delete same_state_node->second;
					same_state_node->second = new_node;
				}

			} // for: jump
		} // for: previous_hyp

		// Retrieve the lowest score of the hypotheses in the current beam
		double best_costs = min_element(beams[beam_index].begin(), beams[beam_index].end(), CompareScore())->second->score;
		double upper_bound = best_costs + pruning_threshold;

		// Apply pruning criterion
		// Yes, removing elements of maps by value is this complicated
		// TODO: Use an alternative data structure to map for the beams
		BeamIterator it = beams[beam_index].begin();
		while ((it = std::find_if(it, beams[beam_index].end(), [&upper_bound](const NodeScore& val){
																													 	 return val.second->score > upper_bound;
																													 })) != beams[beam_index].end()) {
			delete it->second;
			beams[beam_index].erase(it++);
		}

	} // for: beam_index


	// Linear search to find the highest state reached in search
	// It is possible that the final automaton state has not been reached
	// TODO: Use std::map::find()
	// TODO: Maybe return a very high cost at this point, instead of using a smaller state
	StateIdx highest_state_idx = 0;
	for (BeamIterator hyp = beams[n_features-1].begin();
					hyp != beams[n_features-1].end();
					hyp++) {
		highest_state_idx = std::max(highest_state_idx, hyp->first);
	}

	// extract the best hypothesis from the last beam
	Node  *best_node  = beams[n_features-1].find(highest_state_idx)->second;
	double best_costs = best_node->score;

	// backtrack to the initial node and set the alignment
	AlignmentIter align_iter = align_end-1;
	while (best_node->antecessor != NULL) {
		// set the mapping
		(*align_iter)[0].state  = reference[best_node->state];
		(*align_iter)[0].weight = 1;
    (*align_iter)[0].count  = 1;

		align_iter--;

		// set the node for backtracking
		best_node = best_node->antecessor;
	}

	// set the mapping of the first vector to the beginning of the alignment
	(*align_begin)[0].state = reference[0];
	(*align_begin)[0].weight = 1;
	(*align_begin)[0].count = 1;

	// clean the pointers to the nodes used in the beam search
	for (size_t beam_index = 0; beam_index < n_features; beam_index++) {
			for (BeamIterator hyp = beams[beam_index].begin();
					hyp != beams[beam_index].end();
					hyp++) {
				delete hyp->second;
			}
	}

	return best_costs;
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

  /*
  std::cout << "Read alignment: " << std::endl;
  unsigned counter = 0;
  Alignment::iterator it = alignment.begin();

  while (it != alignment.end()) {
    std::cout << counter << " " << (*it).state << std::endl;
    counter++;
    it++;
  }
  */
}

/*****************************************************************************/
