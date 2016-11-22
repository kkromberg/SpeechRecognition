/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <limits>
#include <iostream>

#include "MarkovAutomaton.hpp"
#include "Mixtures.hpp"
#include "TdpModel.hpp"

class Aligner {
public:
  Aligner(MixtureModel const& mixtures, TdpModel const& tdp_model, size_t max_aligns);

  double align_sequence_full(FeatureIter feature_begin, FeatureIter feature_end,
                             MarkovAutomaton const& reference,
                             AlignmentIter align_begin, AlignmentIter align_end);
  double align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                               MarkovAutomaton const& reference,
                               AlignmentIter align_begin, AlignmentIter align_end,
                               double pruning_threshold);
  double align_sequence_fwdbwd(FeatureIter feature_begin, FeatureIter feature_end,
                             MarkovAutomaton const& reference,
                             AlignmentIter align_begin, AlignmentIter align_end);

private:
  MixtureModel const& mixtures_;
  TdpModel     const& tdp_model_;

  size_t max_aligns_;

  typedef std::vector<double>     CostColumn;
  typedef std::vector<CostColumn> CostMatrix;

  CostMatrix forward_pass (FeatureIter feature_begin, FeatureIter feature_end,
	 	 	 	 	 	 	 	 	 	 	 	 	MarkovAutomaton const& reference);
  CostMatrix backward_pass(FeatureIter feature_begin, FeatureIter feature_end,
	 	 	 	 	 	 	 	 	 	 	 	 	MarkovAutomaton const& reference);
  double     path_probability(const CostMatrix& forward_matrix, const CostMatrix& backward_matrix,
  														StateIdx feature, StateIdx state_idx);
	void weighted_alignment_mapping(const FeatureIter& feature_begin,
			const AlignmentIter& align_begin, const AlignmentIter& align_end,
			const CostMatrix& path_probability_matrix);
};

void dump_alignment (std::ostream& out, Alignment const& alignment, size_t  max_aligns);
void write_alignment(std::ostream& out, Alignment const& alignment, size_t  max_aligns);
void read_alignment (std::istream& in,  Alignment&       alignment, size_t& max_aligns);

#endif /* __ALIGNMENT_HPP__ */
