#ifndef _TEACHING_SEARCH_INTERFACE_HH
#define _TEACHING_SEARCH_INTERFACE_HH

#include "Lexicon.hh"
#include "Types.hh"
#include <Search/Search.hh>
#include <Am/AcousticModel.hh>
#include <Lm/ScaledLanguageModel.hh>

// Forward declarations
namespace Speech { class ModelCombination; }

namespace Teaching
{
    /**
     * Interface between decoders in module Teaching
     * and Sprint classes
     */
    class SearchInterface : public Search::SearchAlgorithm
    {
    public:
	struct TracebackItem
	{
	    Word  word;
	    Score score;
	    Time  time;
	    TracebackItem(Word w, Score s, Time t)
		: word(w), score(s), time(t) {}
	};
	typedef std::vector<TracebackItem> Traceback;

    public:
	SearchInterface(const Core::Configuration &);
	virtual ~SearchInterface();

	/* Interface to Search::SearchAlgorithm */
	virtual bool setModelCombination(const Speech::ModelCombination &modelCombination);
	virtual void setGrammar(Fsa::ConstAutomatonRef);
	virtual void restart();
	virtual void feed(const Mm::FeatureScorer::Scorer&);
	virtual void getPartialSentence(Search::SearchAlgorithm::Traceback&);
	virtual void getCurrentBestSentence(Search::SearchAlgorithm::Traceback&) const;
	virtual Lattice::ConstWordLatticeRef getCurrentWordLattice() const;
	virtual void resetStatistics();
	virtual void logStatistics() const;

    protected:
	/* Interface to the implemented decoder */

	/**
	 * initialize / reset decoder.
	 * called for each segment, before processFrame()
	 */
	virtual void initialize() = 0;

	/**
	 * process feature vector at time t.
	 * called for each timeframe
	 * @param t current time (1 .. T)
	 */
	virtual void processFrame(Time t) = 0;

	/**
	 * request current best word sequence
	 * called for each segment, after processFrame()
	 * @param result traceback of best word sequence
	 */
	virtual void getResult(Traceback &result) const = 0;

    protected:
	/**
	 * the log likelihood of the current observation
	 * for mixture model m
	 * @param  m   mixture index
	 * @return log likelihood
	 */
	Score getLogLikelihoodForMixture(Mixture m) const;

	/**
	 * language model log probability of word sequence words.
	 * @param words word sequence, current word first
	 * @return log p( words[0] | words[n-1] words[n] .. word[1] )
	 */
	Score getLanguageModelScore(WordSequence words) const;

	/**
	 * bigram languagemodel log probability of word w given word h
	 * @param w word
	 * @param h history
	 * @return log p(w | h)
	 */
	 Score getLanguageModelScore(Word w, Word h) const;

	 /**
	  * time distortion penalty
	  * @param isSilence state transition from a silence state?
	  * @param distance  length of the transition (0/1/2)
	  * @return score
	  */
	 Score getTransitionScore(bool isSilence, State distance) const;


    public:
	 class Scorer;
	 class AcousticModelScorer;
	 class LanguageModelScorer;
	 class TransitionModelScorer;

    protected:
	Core::Ref<const Am::AcousticModel>       acousticModel_;
	Core::Ref<const Lm::ScaledLanguageModel> lm_;
	Lexicon *lexicon_;

	Score lmScale_;
	Score pronunciationScale_;

    private:
	Mm::FeatureScorer::Scorer currentScorer_;
	Time currentTime_;

    };

    /**
     * base class for scorers
     */
    class SearchInterface::Scorer
    {
    public:
	Scorer(SearchInterface *parent) :
	    parent_(parent) {}
    protected:
	SearchInterface *parent_;
    };

    /**
     * function object to enable access to
     * SearchInterface::getLogLikelihoodForMixture
     */
    class SearchInterface::AcousticModelScorer :
	public SearchInterface::Scorer
    {
    public:
	AcousticModelScorer(SearchInterface *p) :
	    Scorer(p) {}

	Score operator()(Mixture m) const {
	    return parent_->getLogLikelihoodForMixture(m);
	}
    };

    /**
     * function object to enable access to
     * SearchInterface::getLanguageModelScore
     */
    class SearchInterface::LanguageModelScorer :
	public SearchInterface::Scorer
    {
    public:
	LanguageModelScorer(SearchInterface *p) :
	    Scorer(p) {}

	Score operator()(Word w, Word h) const {
	    return parent_->getLanguageModelScore(w, h);
	}
    };

    /**
     * function object to enable access to
     * SearchInterface::getTransitionScore
     */
    class SearchInterface::TransitionModelScorer :
	public SearchInterface::Scorer
    {
    public:
	TransitionModelScorer(SearchInterface *p) :
	    Scorer(p) {}

	Score operator()(bool isSilence, State skip) const {
	    return parent_->getTransitionScore(isSilence, skip);
	}

	enum { exitPenaltyIndex = Am::StateTransitionModel::exit };
    };

}

#endif // _TEACHING_SEARCH_INTERFACE
