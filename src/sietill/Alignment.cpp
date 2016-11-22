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

  double score_to_prob(double score) {
  	return exp(-1 * score);
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
  return 0.0;
}

/*****************************************************************************/

double Aligner::align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                                      MarkovAutomaton const& reference,
                                      AlignmentIter align_begin, AlignmentIter align_end,
                                      double pruning_threshold) {
  // TODO: implement
  return 0.0;
}

Aligner::CostMatrix Aligner::forward_pass(FeatureIter feature_begin, FeatureIter feature_end,
														 	 	 	 	 	 	  MarkovAutomaton const& reference) {
	size_t n_features = feature_end - feature_begin;

	// N+1 x S matrix, the very first column stores the initial costs of the function
	CostMatrix forward_cost_matrix = std::vector<CostColumn>(n_features + 1);
	for (size_t i = 0; i < n_features + 1; i++) {
		forward_cost_matrix[i]    = CostColumn(reference.num_states(), 0.0);
		forward_cost_matrix[i][0] = 0.0;
	}

	// forward pass
	size_t feature_index = 1;
	double current_score = 0.0;
	for (FeatureIter feature_iterator = feature_begin;
			 feature_iterator != feature_end;
			 feature_iterator++, feature_index++) {

		for (StateIdx state_idx = 0; state_idx < reference.num_states(); state_idx++) {

			// get the emission probability scores
			current_score = -1 * mixtures_.score(feature_iterator, reference[state_idx]);

			for (StateIdx prev_state_idx = std::max(0, (int)state_idx - 2);
					 prev_state_idx < state_idx; prev_state_idx++) {

				// accumulate the transition probability from the previous states
				double recursive_transition_probability = tdp_model_.score(reference[state_idx],
																																	 state_idx - prev_state_idx);
				recursive_transition_probability += forward_cost_matrix[feature_index - 1][prev_state_idx];

				// update the score
				current_score = logsum(current_score, -1 * recursive_transition_probability);
			}

			forward_cost_matrix[feature_index][state_idx] = current_score;
		}
	}

	return forward_cost_matrix;
}

Aligner::CostMatrix Aligner::backward_pass(FeatureIter feature_begin, FeatureIter feature_end,
																					 MarkovAutomaton const& reference) {
	size_t n_features = feature_end - feature_begin;
	size_t n_states   = reference.num_states();

	// N+1 x S matrix, the very last column stores the initial costs of the function
	CostMatrix backward_cost_matrix = std::vector<CostColumn>(n_features + 1);
	for (size_t i = 0; i < n_features + 1; i++) {
		backward_cost_matrix[i] = CostColumn(reference.num_states(), 0.0);
	}

	size_t feature_index = n_features;
	double current_score = 0.0;
	for (FeatureIter feature_iterator = feature_end - 1;
			 feature_iterator >= feature_begin;
			 feature_iterator--, feature_index--) {

		for (int state_idx = reference.num_states(); state_idx >= 0; state_idx--) {

			double sum_score = 0.0;
			for (StateIdx next_state_idx = state_idx;
					 next_state_idx < n_states && next_state_idx - state_idx <= 2;
					 next_state_idx++) {
				// accumulate the transition probability from the previous states
				current_score  = -1 * mixtures_ .score(feature_iterator - 1, reference[state_idx] - 1);
				current_score += -1 * tdp_model_.score(reference[state_idx], state_idx - next_state_idx);
				current_score += -1 * backward_cost_matrix[feature_index+1][next_state_idx];

				// update the score
				sum_score = logsum(sum_score, -1 * current_score);
			}

			backward_cost_matrix[feature_index][state_idx] = -1 * current_score;
		}
	}

	return backward_cost_matrix;
}

void Aligner::weighted_alignment_mapping(const FeatureIter& feature_begin,
		const AlignmentIter& align_begin, const AlignmentIter& align_end,
		const CostMatrix& path_probability_matrix) {

	FeatureIter feature_iter = feature_begin;
	size_t feature_idx = 0;
	for (AlignmentIter align_iter = align_begin; align_iter != align_end;
			align_iter++, feature_idx++, feature_iter++) {

		for (size_t n_align = 0; n_align < (*align_iter)->count; n_align++) {
			(*align_iter)[n_align].weight +=
					path_probability_matrix[feature_idx][n_align];
		}

	}

}

double Aligner::align_sequence_fwdbwd(FeatureIter feature_begin, FeatureIter feature_end,
																			MarkovAutomaton const& reference,
																			AlignmentIter align_begin, AlignmentIter align_end) {
	const CostMatrix forward_cost_matrix  = forward_pass (feature_begin, feature_end, reference);
	const CostMatrix backward_cost_matrix = backward_pass(feature_begin, feature_end, reference);

	const size_t n_features = feature_end - feature_begin;
	const size_t n_states   = reference.num_states();

	CostMatrix path_probability_matrix = std::vector<CostColumn>(n_features);
	for (size_t i = 0; i < n_features; i++) {
		path_probability_matrix[i] = CostColumn(n_states, 0.0);
	}

	double path_score = 0.0;
	for (size_t feature_idx = 0; feature_idx < n_features; feature_idx++) {
		for (size_t state_idx = 0; state_idx < n_states; state_idx++) {
			path_probability_matrix[feature_idx][state_idx] =
					path_probability(forward_cost_matrix, backward_cost_matrix, feature_idx, state_idx);
		}

		// get the score of the whole path before normalizing the entries
		if (feature_idx == n_features - 1) {
			path_score = path_probability_matrix[feature_idx][n_states-1];
		}

		// Normalize path probabilties for each feature ( in log space)
		double log_normalization = std::accumulate(path_probability_matrix[feature_idx].begin(),
										path_probability_matrix[feature_idx].end(),
										0.0);

		std::transform(path_probability_matrix[feature_idx].begin(),
									 path_probability_matrix[feature_idx].end(),
									 path_probability_matrix[feature_idx].begin(),
									 std::bind2nd(std::minus<double>(), log_normalization));

		std::transform(path_probability_matrix[feature_idx].begin(),
									 path_probability_matrix[feature_idx].end(),
									 path_probability_matrix[feature_idx].begin(),
									 score_to_prob);

		// Sort the probabilities in descending order for an easy access to the probabilities
		std::sort(path_probability_matrix[feature_idx].begin(), path_probability_matrix[feature_idx].end(),
							std::greater<double>());
	}

	weighted_alignment_mapping(feature_begin, align_begin, align_end, path_probability_matrix);

	return path_score;
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

double Aligner::path_probability(const CostMatrix& forward_matrix,
		const CostMatrix& backward_matrix, StateIdx feature_index, StateIdx state_idx) {
	double score = forward_matrix[feature_index+1][state_idx];
	score += backward_matrix[feature_index][state_idx];

	return -1 * score;
}
/*****************************************************************************/
