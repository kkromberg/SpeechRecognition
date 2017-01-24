#ifndef WORDCONDITIONEDTREESEARCH_HH_
#define WORDCONDITIONEDTREESEARCH_HH_

#include <vector>
#include "SearchInterface.hh"

namespace Teaching {

// -------------------------------------------------------------------------------------------------

typedef unsigned int Arc;

static const Index onlyStart = std::numeric_limits<Index>::max() - 1;

static const Arc invalidArc = std::numeric_limits<Arc>::max();

/** hidden markov model state */
typedef short int HmmState;

static const HmmState inactiveState = std::numeric_limits<HmmState>::max();

// -------------------------------------------------------------------------------------------------

class WordConditionedTreeSearch : public SearchInterface {
public:
    struct WordHypothesis {
	Word word;
	Score score;
	Index backpointer;

	WordHypothesis(Word word, Score score, Index backpointer) :
	    word(word), score(score), backpointer(backpointer) {
	}

	bool operator<(const WordHypothesis &other) const {
	    return score < other.score;
	}
    };

    WordConditionedTreeSearch(const Core::Configuration &config);
    virtual ~WordConditionedTreeSearch();

	virtual bool setModelCombination(const Speech::ModelCombination &modelCombination);

protected:
    /**
     * Initialize word-conditioned tree search.
     */
    virtual void initialize();

    /**
     * Recognize one time frame.
     */
    virtual void processFrame(Time t);

    /**
     * Return best found word sequence.
     */
    virtual void getResult(Traceback &result) const;

private:
    class SearchSpace;
    class ActiveTrees;

    /**
     * Create tree lexicon.
     */
    void buildTreeLexicon();

    SearchSpace *searchSpace_;
    AcousticModelScorer *amScorer_;
    LanguageModelScorer *lmScorer_;

    /** score offset relative to best state hypothesis for acoustic pruning */
    Score acousticPruningThreshold_;

    static const Core::ParameterFloat paramAcousticPruningThreshold_;
    static const Core::ParameterInt paramAcousticPruningLimit_;
};

};

#endif // WORDCONDITIONEDTREESEARCH_HH_
