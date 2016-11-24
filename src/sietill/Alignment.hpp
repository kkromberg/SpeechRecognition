/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

#include <map>
#include <limits>
#include <iostream>
#include <utility>

#include "MarkovAutomaton.hpp"
#include "Mixtures.hpp"
#include "TdpModel.hpp"

class Aligner {
public:

	struct Node {
		Node     *antecessor;
		StateIdx state;
		double   score;

		Node(Node *a, StateIdx s, double c) :
			antecessor(a), state(s), score(c) {}
	};

	typedef std::pair<StateIdx, Node*> NodeScore;
	struct CompareScore
	{
	    bool operator()(const NodeScore& left, const NodeScore& right) const
	    {
	        return left.second->score < right.second->score;
	    }
	};

	typedef std::map<StateIdx, Node*> Beam;
	typedef std::map<StateIdx, Node*>::iterator BeamIterator;

	// Data structures for a 2D alignment using dynamic programming
	typedef std::vector< std::vector<double> >    CostMatrix;
	typedef std::vector< std::vector<int> > 	 BackpointerMatrix;
  Aligner(MixtureModel const& mixtures, TdpModel const& tdp_model, size_t max_aligns);

  double compute_local_costs(StateIdx state, double feature);
  double align_sequence_full(FeatureIter feature_begin, FeatureIter feature_end,
                             MarkovAutomaton const& reference,
                             AlignmentIter align_begin, AlignmentIter align_end);
  double align_sequence_pruned(FeatureIter feature_begin, FeatureIter feature_end,
                               MarkovAutomaton const& reference,
                               AlignmentIter align_begin, AlignmentIter align_end,
                               double pruning_threshold);

private:
  MixtureModel const& mixtures_;
  TdpModel     const& tdp_model_;

  size_t max_aligns_;
};

void dump_alignment (std::ostream& out, Alignment const& alignment, size_t  max_aligns);
void write_alignment(std::ostream& out, Alignment const& alignment, size_t  max_aligns);
void read_alignment (std::istream& in,  Alignment&       alignment, size_t& max_aligns);

#endif /* __ALIGNMENT_HPP__ */
