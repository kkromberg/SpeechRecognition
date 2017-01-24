#ifndef TEACHING_LINEAR_SEARCH_HH
#define TEACHING_LINEAR_SEARCH_HH

#include <Teaching/SearchInterface.hh>

namespace Teaching
{

    class LinearSearch :
	public SearchInterface
    {
    public:
	LinearSearch(const Core::Configuration &c);
	virtual ~LinearSearch();

	virtual bool setModelCombination(const Speech::ModelCombination &modelCombination);

    protected:
	virtual void initialize();
	virtual void processFrame(Time t);
	virtual void getResult(Traceback &result) const;

    protected:

	class SearchSpace;

	struct WordBoundaryHypothesis
	{
	    Word  word;
	    Score score;
	    Index backpointer;

	    WordBoundaryHypothesis(Word w, Score s, Index b) :
		word(w), score(s), backpointer(b) {}

	    WordBoundaryHypothesis() :
		word(invalidWord), score(maxScore), backpointer(invalidIndex) {}

	    bool operator<(const WordBoundaryHypothesis &o) const {
		return score < o.score;
	    }
	};

    protected:

	typedef std::vector<MixtureSequence> LinearLexicon;
	void buildLinearLexicon(LinearLexicon &lexicon);
	LinearLexicon linearLexicon_;

	typedef std::vector<WordBoundaryHypothesis> WordBoundaryHypotheses;
	WordBoundaryHypotheses wordEndHypotheses_, wordStartHypotheses_;

	Score languageModelPruningThreshold_, acousticPruningThreshold_;

	SearchSpace *searchSpace_;
	AcousticModelScorer *amScorer_;
	LanguageModelScorer *lmScorer_;

    private:
	static const Core::ParameterFloat paramAcousticPruningThreshold;
	static const Core::ParameterFloat paramLmPruningThreshold;
    };

};

#endif // TEACHING_LINEAR_SEARCH_HH
