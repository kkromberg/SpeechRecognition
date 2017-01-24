#include "LinearSearch.hh"
#include "BookKeeping.hh"
#include <Speech/ModelCombination.hh>

using namespace Teaching;

class LinearSearch::SearchSpace
{
private:
    struct StateHypothesis
    {
	State state;
	Score score;
	Index backpointer;

	StateHypothesis(State st, Score sc, Index bp) :
	    state(st), score(sc), backpointer(bp) {}

	StateHypothesis() :
	    state(0), score(maxScore), backpointer(invalidIndex) {}
    };

    struct WordHypothesis
    {
	Word  word;
	Index stateHypBegin, stateHypEnd;
	Index entryStateHypothesis;

	WordHypothesis(Word w, Index sb, Index se) :
	    word(w), stateHypBegin(sb), stateHypEnd(se),
	    entryStateHypothesis(invalidIndex) {}

	WordHypothesis() :
	    word(invalidWord),
	    stateHypBegin(invalidIndex), stateHypEnd(invalidIndex),
	    entryStateHypothesis(invalidIndex) {}
    };

    typedef std::vector<WordHypothesis>             WordHypotheses;
    typedef std::vector<StateHypothesis>            StateHypotheses;
    typedef std::vector<StateHypotheses::size_type> StateHypothesesMap;
    typedef std::vector<WordHypotheses::size_type>  WordHypothesesMap;

public:
    SearchSpace(const LinearLexicon&, Word silence);
    ~SearchSpace();

    /**
     * read transition scores
     */
    void setTransitionScores(const TransitionModelScorer &scorer);
    /**
     * reset before new segment
     */
    void reset();
    /**
     * add first hypotheses to initialize the search space
     */
    void addInitialHypothesis(WordBoundaryHypotheses &wordEndHyps);
    /**
     * create hypotheses for starting words using the word end hypothesees in wordEndHyps.
     * language model probability is calculated for starting words,
     * word start hypotheses are written to wordStartHyps
     */
    void bigramRecombination(const WordBoundaryHypotheses &wordEndHyps, WordBoundaryHypotheses &wordStartHyps,
			     const LanguageModelScorer &scorer) const;
    /**
     * create word hypotheses for the word start hypotheses in wordStartHyps.
     * wordStartHypotheses are pruned using the language model pruning threshold pruningThreshold
     */
    void insertWordStartHypotheses(const WordBoundaryHypotheses &wordStartHyps, Score pruningThreshold);
    /**
     * expand current state hypotheses
     */
    void expandHypotheses();
    /**
     * add acoustic scores for the current feature vector to all state hypotheses
     */
    void addAcousticScores(const AcousticModelScorer &featureScore);
    /**
     * prune state hypotheses and find word end hypotheses.
     */
    void pruneStatesAndFindWordEnds(WordBoundaryHypotheses &wordEndHyps, Score threshold);
    /**
     * merge silence copies and corresponding word ends
     */
    void mergeSilenceToBigramNodes(WordBoundaryHypotheses &wordEndHyps) const;
    /**
     * add book keeeping entries for word end hypotheses
     */
    void addBookKeepingEntries(Time t, WordBoundaryHypotheses &wordEndHyps);
    /**
     * traceback of the best word end hypotheses in wordEndHyps
     */
    void traceback(const WordBoundaryHypotheses &wordEndHyps, Traceback &result) const;
    /**
     * currently best score of all state hypotheses
     */
    Score currentBestScore() const { return currentBestScore_; }

private:
    void addEntryStateHypothesis(Word word, Score score, Index backpointer);
    void expandWordHypothesis(Index wordHypIndex);
    void expandState(Word word, const StateHypothesis &stateHyp);


    /**
     * map silence copy to corresponding word
     */
    Word mapSilenceCopy(Word w) const;
    /**
     * silence copy for word w
     */
    Word silenceCopy(Word w) const;
    /**
     * map silence copies to silence
     */
    Word acousticWord(Word w) const;

    /**
     * silence or silence copy
     */
    bool isSilence(Word w) const;

private:
    StateHypotheses    stateHypotheses_,
		       newStateHypotheses_;
    WordHypotheses     wordHypotheses_;
    StateHypothesesMap stateHypothesisMap_;
    WordHypothesesMap  wordHypothesisMap_;
    std::vector< std::vector<Score> > transitionScores_;

    BookKeeping *bookKeeping_;
    const LinearLexicon &lexicon_;

    Index currentFirstNewStateHypothesis_;
    Score currentBestScore_;
    Index nWords_;
    Word silence_;
    static const State maxSkip_;
    static const Time  purgeStorageInterval_;
};

const State LinearSearch::SearchSpace::maxSkip_              = 2;
const Time  LinearSearch::SearchSpace::purgeStorageInterval_ = 50;

LinearSearch::SearchSpace::SearchSpace(const LinearLexicon &lexicon, Word silence) :
    bookKeeping_(new BookKeeping),
    lexicon_(lexicon),
    nWords_(lexicon.size()),
    silence_(silence)
{
    wordHypothesisMap_.resize(nWords_ * 2);        // words + silence copies
    Index maxWordLength = 0;
    for (Index w = 0; w < lexicon.size(); ++w)
	if (lexicon[w].size() > maxWordLength)
	    maxWordLength = std::max(int(maxWordLength), int(lexicon[w].size()));
    stateHypothesisMap_.resize(maxWordLength + 1); // states + virtual entry state
}

LinearSearch::SearchSpace::~SearchSpace()
{
    delete bookKeeping_;
}

void LinearSearch::SearchSpace::setTransitionScores(const TransitionModelScorer &transitionScore)
{
    transitionScores_.resize(2);
    require(TransitionModelScorer::exitPenaltyIndex > maxSkip_);
    for (int isSilence = 0; isSilence <= 1; ++isSilence) {
	transitionScores_[isSilence].resize(maxSkip_ + 2);
	for(int s = 0; s <= maxSkip_; ++s) {
	    transitionScores_[isSilence][s] = transitionScore(isSilence, s);
	}
	transitionScores_[isSilence][maxSkip_ + 1] = transitionScore(isSilence, TransitionModelScorer::exitPenaltyIndex);
    }
}

void LinearSearch::SearchSpace::reset()
{
    require(transitionScores_.size());
    wordHypotheses_.clear();
    stateHypotheses_.clear();
    newStateHypotheses_.clear();
    bookKeeping_->clear();
    currentBestScore_ = maxScore;
    std::fill(wordHypothesisMap_.begin(), wordHypothesisMap_.end(), invalidIndex);
    std::fill(stateHypothesisMap_.begin(), stateHypothesisMap_.end(), invalidIndex);
}

inline Word LinearSearch::SearchSpace::mapSilenceCopy(Word w) const
{
    return (w == silence_ ? w : w % nWords_);
}

inline Word LinearSearch::SearchSpace::silenceCopy(Word w) const
{
    return (w == silence_ ? w : w + nWords_);
}

inline Word LinearSearch::SearchSpace::acousticWord(Word w) const
{
    return (w < nWords_ ? w : silence_);
}

inline bool LinearSearch::SearchSpace::isSilence(Word w) const
{
    return (w == silence_ || w >= nWords_);
}

void LinearSearch::SearchSpace::addInitialHypothesis(WordBoundaryHypotheses &wordEndHyps)
{
    // startup word hypothesis
    wordEndHyps.push_back( WordBoundaryHypothesis(silence_, 0, 0) );
    addBookKeepingEntries(0, wordEndHyps);
}


void LinearSearch::SearchSpace::bigramRecombination(const WordBoundaryHypotheses &wordEndHyps,
						    WordBoundaryHypotheses &wordStartHyps,
						    const LanguageModelScorer &lmScore) const
{
    wordStartHyps.clear();
    wordStartHyps.resize(nWords_);

    for (WordBoundaryHypotheses::const_iterator we = wordEndHyps.begin(); we != wordEndHyps.end(); ++we) {

	Word prevWord = mapSilenceCopy(we->word);

	// transition from previous word or silence to new word
	for (Word w = 0; w < nWords_; ++w) {
	    if (w == silence_) continue; // no transition to silence

	    Score newScore = we->score + lmScore(w, prevWord);
	    if (newScore < wordStartHyps[w].score) {
		wordStartHyps[w] = WordBoundaryHypothesis(w, newScore, we->backpointer);
	    }
	}
	// transition from previous word to silence copy
	if(we->word < nWords_) { // exclude silence copies
	    wordStartHyps.push_back( WordBoundaryHypothesis(silenceCopy(we->word), we->score, we->backpointer) );
	}
    }
}

void LinearSearch::SearchSpace::insertWordStartHypotheses(const WordBoundaryHypotheses &wordStartHyps,
							  Score pruningThreshold)
{

    for (WordBoundaryHypotheses::const_iterator ws = wordStartHyps.begin(); ws != wordStartHyps.end(); ++ws) {
	if (ws->score < pruningThreshold) {
	    addEntryStateHypothesis(ws->word, ws->score, ws->backpointer);
	}
    }
}

void LinearSearch::SearchSpace::addEntryStateHypothesis(Word word, Score score, Index backpointer)
{
    verify(word < wordHypothesisMap_.size());
    if (wordHypothesisMap_[word] == invalidIndex) {
	// activate word hypothesis
	wordHypothesisMap_[word] = wordHypotheses_.size();
	wordHypotheses_.push_back( WordHypothesis(word, invalidIndex, invalidIndex) );
    }
    verify(wordHypothesisMap_[word] < wordHypotheses_.size());
    wordHypotheses_[ wordHypothesisMap_[word] ].entryStateHypothesis = stateHypotheses_.size();
    stateHypotheses_.push_back( StateHypothesis(0, score, backpointer) );
}

void LinearSearch::SearchSpace::expandHypotheses()
{
    newStateHypotheses_.clear();
    for (Index w = 0; w < wordHypotheses_.size(); ++w) {
	expandWordHypothesis(w);
    }
    stateHypotheses_.swap(newStateHypotheses_);
}

void LinearSearch::SearchSpace::expandWordHypothesis(Index wordHypIndex)
{
    WordHypothesis &wordHyp = wordHypotheses_[wordHypIndex];
    currentFirstNewStateHypothesis_ = newStateHypotheses_.size();

    if(wordHyp.entryStateHypothesis != invalidIndex) {
	expandState(wordHyp.word, stateHypotheses_[wordHyp.entryStateHypothesis]);
	wordHyp.entryStateHypothesis = invalidIndex;
    }

    for (Index s = wordHyp.stateHypBegin; s < wordHyp.stateHypEnd; ++s)
	expandState(wordHyp.word, stateHypotheses_[s]);
    wordHyp.stateHypBegin = currentFirstNewStateHypothesis_;
    wordHyp.stateHypEnd   = newStateHypotheses_.size();

}

void LinearSearch::SearchSpace::expandState(Word word, const StateHypothesis &stateHyp)
{
    const MixtureSequence &mixtures = lexicon_[acousticWord(word)];
    for (State successorState = std::max(State(1), stateHyp.state);
	 successorState <= std::min(State(stateHyp.state + maxSkip_), State(mixtures.size()));
	 ++successorState) {

	Score newScore = stateHyp.score;
	State timeDistortion = (successorState - stateHyp.state);
	// don't add transition score for the transition from the virtual entry state
	// to the first state
	if (stateHyp.state || timeDistortion > 1) {
	    newScore += transitionScores_[isSilence(word)][timeDistortion];
	}
	Index stateHypIndex = stateHypothesisMap_[successorState];
	if (stateHypIndex < currentFirstNewStateHypothesis_ ||
	    stateHypIndex >= newStateHypotheses_.size()     ||
	    newStateHypotheses_[stateHypIndex].state != successorState) {
	    // state hypotheses for state successorState does not exist -> create it
	    stateHypothesisMap_[successorState] = newStateHypotheses_.size();
	    newStateHypotheses_.push_back( StateHypothesis(successorState, newScore, stateHyp.backpointer) );
	} else {
	    // another hypothesis for state sucessorState exists -> recombine
	    StateHypothesis &newStateHyp = newStateHypotheses_[stateHypIndex];
	    if (newStateHyp.score >= newScore) {
		newStateHyp.score = newScore;
		newStateHyp.backpointer = stateHyp.backpointer;
	    }
	}
    }
}

void LinearSearch::SearchSpace::addAcousticScores(const AcousticModelScorer &featureScore)
{
    currentBestScore_ = maxScore;
    for (Index w = 0; w < wordHypotheses_.size(); ++w) {
	Word word = acousticWord(wordHypotheses_[w].word);
	for (Index s = wordHypotheses_[w].stateHypBegin; s < wordHypotheses_[w].stateHypEnd; ++s) {
	    stateHypotheses_[s].score += featureScore( lexicon_[word][stateHypotheses_[s].state-1] );
	    if (stateHypotheses_[s].score < currentBestScore_)
		currentBestScore_= stateHypotheses_[s].score;
	}
    }
}

void LinearSearch::SearchSpace::pruneStatesAndFindWordEnds(WordBoundaryHypotheses &wordEndHyps, Score threshold)
{
    wordEndHyps.clear();
    WordHypotheses::iterator  wordHypIn, wordHypOut, wordHypBegin;
    StateHypotheses::iterator stateHypIn, stateHypOut, stateHypBegin, stateHypEnd;
    wordHypBegin = wordHypotheses_.begin();
    stateHypIn = stateHypOut = stateHypBegin = stateHypotheses_.begin();

    for (wordHypIn = wordHypOut = wordHypotheses_.begin(); wordHypIn != wordHypotheses_.end(); ++wordHypIn) {
	verify(stateHypIn == stateHypBegin + wordHypIn->stateHypBegin);
	wordHypIn->stateHypBegin = stateHypOut - stateHypBegin;
	State nStates = lexicon_[acousticWord(wordHypIn->word)].size();
	for (stateHypEnd = stateHypBegin + wordHypIn->stateHypEnd; stateHypIn < stateHypEnd; ++stateHypIn) {
	    Score score = stateHypIn->score + transitionScores_[isSilence(wordHypIn->word)][maxSkip_ + 1]; // add exit penalty
	    if (score < threshold) {
		*(stateHypOut++) = *stateHypIn;
		if (stateHypIn->state == nStates) {
		    // found a word end
		    wordEndHyps.push_back( WordBoundaryHypothesis(wordHypIn->word, score, stateHypIn->backpointer) );
		}
	    }
	}
	// remove word hypotheses without state hypotheses
	wordHypIn->stateHypEnd = stateHypOut - stateHypBegin;
	if (wordHypIn->stateHypEnd - wordHypIn->stateHypBegin > 0) {
	    // keep word hypothesis
	    wordHypothesisMap_[wordHypIn->word] = wordHypOut - wordHypBegin;
	    *(wordHypOut++) = *wordHypIn;
	} else {
	    // deactivate word hypothesis
	    wordHypothesisMap_[wordHypIn->word] = invalidIndex;
	}
    }
    wordHypotheses_.erase(wordHypOut, wordHypotheses_.end());
    stateHypotheses_.erase(stateHypOut, stateHypotheses_.end());
}

void LinearSearch::SearchSpace::mergeSilenceToBigramNodes(WordBoundaryHypotheses &wordEndHyps) const
{
    std::vector<Index> wordEnds(nWords_, invalidIndex);
    Index nWordEnds = 0;
    for (Index we = 0; we < wordEndHyps.size(); ++we) {
	Word acuWord = wordEndHyps[we].word;
	Word word    = mapSilenceCopy(acuWord);
	if (wordEnds[word] == invalidIndex) {
	    wordEnds[word] = we;
	    ++nWordEnds;
	}
	Index wordIndex = wordEnds[word];
	if (wordEndHyps[we].score <= wordEndHyps[wordIndex].score) {
	    wordEndHyps[wordIndex] = wordEndHyps[we];
	}
    }
    wordEndHyps.erase(wordEndHyps.begin() + nWordEnds, wordEndHyps.end());
}

void LinearSearch::SearchSpace::addBookKeepingEntries(Time t, WordBoundaryHypotheses &wordEndHyps)
{
    if (t % purgeStorageInterval_ == 0) {
	bookKeeping_->tagActiveEntries(t, stateHypotheses_.begin(), stateHypotheses_.end());
    }
    for (WordBoundaryHypotheses::iterator we = wordEndHyps.begin(); we != wordEndHyps.end(); ++we) {
	Index newBackpointer = bookKeeping_->addEntry(acousticWord(we->word), we->score, we->backpointer, t);
	we->backpointer = newBackpointer;

	if (t == 0) {
	    // self loop
	    bookKeeping_->entry(newBackpointer).backpointer = newBackpointer;
	}

	// avoid chains of silence
	if (we->word == silence_) {
	    BookKeeping::Entry &curEntry = bookKeeping_->entry(newBackpointer);
	    if (bookKeeping_->entry(curEntry.backpointer).word == silence_)
		curEntry.backpointer = bookKeeping_->entry(curEntry.backpointer).backpointer;
	}
    }
}

void LinearSearch::SearchSpace::traceback(const WordBoundaryHypotheses &wordEndHyps, Traceback &result) const
{
    result.clear();
    if (wordEndHyps.size() == 0)
	return;
    WordBoundaryHypotheses::const_iterator best = std::min_element(wordEndHyps.begin(), wordEndHyps.end());
    Index backpointer = best->backpointer;
    verify(backpointer < bookKeeping_->size());
    Index cnt = 0;
    while(bookKeeping_->entry(backpointer).time > 0) {
	result.push_back( bookKeeping_->entry(backpointer) );
	backpointer = bookKeeping_->entry(backpointer).backpointer;
	verify(backpointer < bookKeeping_->size());
	verify(++cnt < bookKeeping_->size());
    }
    std::reverse(result.begin(), result.end());
}


// =========================================================================================


const Core::ParameterFloat LinearSearch::paramAcousticPruningThreshold(
    "acoustic-pruning", "acoustic pruning threshold", maxScore, 0.0);

const Core::ParameterFloat LinearSearch::paramLmPruningThreshold(
    "lm-pruning", "language model pruning threshold", maxScore, 0.0);

LinearSearch::LinearSearch(const Core::Configuration &c) :
    Core::Component(c),
    SearchInterface(c),
    searchSpace_(0),
    amScorer_(new AcousticModelScorer(this)),
    lmScorer_(new LanguageModelScorer(this))
{
    languageModelPruningThreshold_ = paramLmPruningThreshold(config);
    if (paramLmPruningThreshold(config) > maxScore)
	languageModelPruningThreshold_ = maxScore;
    acousticPruningThreshold_      = paramAcousticPruningThreshold(config);
    if (paramAcousticPruningThreshold(config) > maxScore)
	acousticPruningThreshold_ = maxScore;
}

LinearSearch::~LinearSearch()
{
    delete searchSpace_;
    delete amScorer_;
    delete lmScorer_;
}

bool LinearSearch::setModelCombination(const Speech::ModelCombination &modelCombination)
{
    SearchInterface::setModelCombination(modelCombination);
    buildLinearLexicon(linearLexicon_);
    searchSpace_ = new SearchSpace(linearLexicon_, lexicon_->silence());
    searchSpace_->setTransitionScores(TransitionModelScorer(this));
    return true;
}


void LinearSearch::buildLinearLexicon(LinearLexicon &mixturesForWord)
{
    mixturesForWord.resize(lexicon_->nWords());
    for (Word w = 0; w < lexicon_->nWords(); ++w) {
	lexicon_->getMixtureSequence(w, mixturesForWord[w]);
    }
}

void LinearSearch::initialize()
{
    verify(searchSpace_);
    wordEndHypotheses_.clear();
    searchSpace_->reset();
    searchSpace_->addInitialHypothesis(wordEndHypotheses_);
}

void LinearSearch::processFrame(Time t)
{
    searchSpace_->bigramRecombination(wordEndHypotheses_, wordStartHypotheses_, *lmScorer_);
    WordBoundaryHypotheses::const_iterator best = std::min_element(wordStartHypotheses_.begin(), wordStartHypotheses_.end());
    Score lmPruningThreshold = languageModelPruningThreshold_;
    if (lmPruningThreshold < maxScore)
	lmPruningThreshold += best->score;
    searchSpace_->insertWordStartHypotheses(wordStartHypotheses_, lmPruningThreshold);

    searchSpace_->expandHypotheses();
    searchSpace_->addAcousticScores(*amScorer_);

    Score acPruningThreshold = acousticPruningThreshold_;
    if (acPruningThreshold < maxScore)
	acPruningThreshold += searchSpace_->currentBestScore();
    searchSpace_->pruneStatesAndFindWordEnds(wordEndHypotheses_, acPruningThreshold);

    searchSpace_->mergeSilenceToBigramNodes(wordEndHypotheses_);
    searchSpace_->addBookKeepingEntries(t, wordEndHypotheses_);
}

void LinearSearch::getResult(Traceback &result) const
{
    searchSpace_->traceback(wordEndHypotheses_, result);
}
