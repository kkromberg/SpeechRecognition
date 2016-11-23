/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Alignment.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

/*****************************************************************************/

namespace {
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
	std::cerr << "Number of features: " << feature_number << std::endl;
	// initialise cost and backpointer matrix
	for (StateIdx state = 0; state < state_number; state++){
		cost_matrix.push_back(std::vector<double>(feature_number, std::numeric_limits<double>::infinity())); // set costs to infinity
		backpointer_matrix.push_back(std::vector<size_t>(feature_number, 0));													  	 // backpointer initially with zeros
	}
	double local_costs = std::numeric_limits<double>::infinity();
	//double local_costs,
	size_t max_state = 2;
	size_t min_state = state_number-1-(feature_number-2)*2;
	size_t t = 1;
	// costs for the first point are fixed
	cost_matrix[0][0] = mixtures_.score(feature_begin, reference[0]);

	for (FeatureIter feature_iter = feature_begin+1; feature_iter != feature_end; feature_iter++,t++,
		min_state += 2, max_state += 2) { // loop features
		//TODO determine slope for the current point
		//double result = tdp_model_.score(0, 0);
		int min = min_state;
		int max = max_state;
		//std::cout<<std::endl<<" ";
		for (StateIdx state = std::max(0, min); state < std::min(state_number-1, max); state++) {
			//std::cout<<"State number "<<state<<std::endl;// loop states
				local_costs = mixtures_.score(feature_iter, reference[state]);
				//std::cout<<state<<std::endl<<" ";
				double best = std::numeric_limits<double>::infinity();
				StateIdx state_prime=0;
				//std::cout<<std::max(0,state_number-max+2)<<std::endl;
				for (size_t i=std::max(0,state-max+2); i<=std::min(state+0,2);i++){
					//std::cout<<"i="<<i<<std::endl;
					double temp=cost_matrix[state-i][t-1] + tdp_model_.score(reference[state-i], i);
					if(temp<best){
						best=temp;
						state_prime=state-i;
						//std::cout<<state_prime<<" ";
					}
				}
				//std::cout<<local_costs + best<<" "<<std::endl<<" ";
				cost_matrix[state][t] = local_costs + best;
				//std::cout<<cost_matrix[state][t]<<" ";
				backpointer_matrix[state][t]=state_prime;
				//std::cout<<backpointer_matrix[state][t]<<" ";

				}
		//std::cout<<std::endl;
		}
	// TODO mapping of automaton states to alignment
	size_t feature_index = feature_number - 1;
			size_t state_index   = state_number - 1;
			for (AlignmentIter align_iter = align_end-1; align_iter != align_begin - 1; align_iter--, feature_index--) { // loop
				StateIdx automaton_state = reference[state_index];
				(*align_iter)->state = automaton_state;
				//std::cout << feature_index << " " << state_index << std::endl;
				state_index = backpointer_matrix[state_index][feature_index];
			}
			// return the cost of best path
	  return cost_matrix[state_number-1][feature_number-1];
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
		std::cerr << "Number of features: " << feature_number << std::endl;
		// initialise cost and backpointer matrix
		for (StateIdx state = 0; state < state_number; state++){
			cost_matrix.push_back(std::vector<double>(feature_number, std::numeric_limits<double>::infinity())); // set costs to infinity
			backpointer_matrix.push_back(std::vector<size_t>(feature_number, 0));													  	 // backpointer initially with zeros
		}
		double local_costs = std::numeric_limits<double>::infinity();
		//double local_costs,
		size_t max_state = 2;
		size_t min_state = state_number-1-(feature_number-2)*2;
		size_t t = 1;
		// costs for the first point are fixed
		cost_matrix[0][0] = mixtures_.score(feature_begin, reference[0]);

		for (FeatureIter feature_iter = feature_begin+1; feature_iter != feature_end; feature_iter++,t++,
			min_state += 2, max_state += 2) { // loop features
			//TODO determine slope for the current point
			//double result = tdp_model_.score(0, 0);
			int min = min_state;
			int max = max_state;
			//std::cout<<std::endl<<" ";
			double min_cost = std::numeric_limits<double>::infinity();
			for (StateIdx state = std::max(0, min); state <= std::min(state_number-1, max); state++) {
				//std::cout<<"State number "<<state<<std::endl;// loop states
					local_costs = mixtures_.score(feature_iter, reference[state]);
					//std::cout<<state<<std::endl<<" ";
					double best = std::numeric_limits<double>::infinity();
					StateIdx state_prime=0;
					//std::cout<<std::max(0,state_number-max+2)<<std::endl;
					for (size_t i=std::max(0,state-max+2); i<=std::min(state+0,2);i++){
						//std::cout<<"i="<<i<<std::endl;
						double temp=cost_matrix[state-i][t-1] + tdp_model_.score(reference[state-i], i);
						if(temp<best){
							best=temp;
							state_prime=state-i;
							//std::cout<<state_prime<<" ";
						}
					}
					//std::cout<<local_costs + best<<" "<<std::endl<<" ";
					cost_matrix[state][t] = local_costs + best;
					if (cost_matrix[state][t]<min_cost) min_cost = cost_matrix[state][t];
					//std::cout<<cost_matrix[state][t]<<" ";
					backpointer_matrix[state][t]=state_prime;
					//std::cout<<backpointer_matrix[state][t]<<" ";

					}
			for (StateIdx state = std::max(0, min); state < std::min(state_number-1, max); state++) {
				if (cost_matrix[state][t]>min_cost+pruning_threshold) cost_matrix[state][t] = std::numeric_limits<double>::infinity();
			}
			//std::cout<<std::endl;
			}
		// TODO mapping of automaton states to alignment
		size_t feature_index = feature_number - 1;
		size_t state_index   = state_number - 1;
		for (AlignmentIter align_iter = align_end-1; align_iter != align_begin - 1; align_iter--, feature_index--) { // loop
			StateIdx automaton_state = reference[state_index];
			(*align_iter)->state = automaton_state;
			//std::cout << feature_index << " " << state_index << std::endl;
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
