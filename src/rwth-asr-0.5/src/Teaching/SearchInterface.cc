#include "SearchInterface.hh"
#include <Speech/ModelCombination.hh>


using namespace Teaching;

/*** SearchInteface ***/

SearchInterface::SearchInterface(const Core::Configuration &c) :
    Core::Component(c),
    SearchAlgorithm(c),
    lexicon_(0),
    lmScale_(1.0),
    pronunciationScale_(1.0) /* @todo add parameter */
{}

SearchInterface::~SearchInterface()
{
    delete lexicon_;
}

bool SearchInterface::setModelCombination(const Speech::ModelCombination &modelCombination)
{
    acousticModel_ = modelCombination.acousticModel();
    lm_ = modelCombination.languageModel();
    lexicon_ = new Lexicon(modelCombination.lexicon(), acousticModel_);
    return true;
}

void SearchInterface::restart()
{
    initialize();
    currentTime_ = 0;
    currentScorer_.reset();
}

void SearchInterface::feed(const Mm::FeatureScorer::Scorer &s)
{
    currentScorer_ = s;
    processFrame(++currentTime_);
}

void SearchInterface::getCurrentBestSentence(Search::SearchAlgorithm::Traceback &result) const
{
    Traceback traceback;
    getResult(traceback);
    result.clear();
    for(Traceback::const_iterator i = traceback.begin(); i != traceback.end(); ++i) {
	result.push_back( Search::SearchAlgorithm::TracebackItem( lexicon_->pronunciation(i->word), i->time,
								  ScoreVector(i->score, 0), Lattice::WordBoundary::Transit()) );
    }
}

/*** functions for the decoder ***/

Score SearchInterface::getLogLikelihoodForMixture(Mixture m) const
{
    require(currentScorer_);
    return currentScorer_->score(m);
}

Score SearchInterface::getLanguageModelScore(WordSequence ws) const
{
    Score score = 0;
    Lm::History history = lm_->startHistory();

    for(WordSequence::size_type i = std::max(ws.size() - 1, (WordSequence::size_type)0); i > 0; --i) {
	Lm::extendHistoryByLemmaPronunciation(lm_, lexicon_->pronunciation(ws[i]), history);
    }
    Lm::addLemmaPronunciationScore(lm_, lexicon_->pronunciation(ws.front()), pronunciationScale_, lmScale_, history, score);
    return score;
}

Score SearchInterface::getLanguageModelScore(Word w, Word h) const
{
    WordSequence ws(2); ws[0] = w; ws[1] = h;
    return getLanguageModelScore(ws);
}

Score SearchInterface::getTransitionScore(bool isSilence, State distance) const
{
    Am::AcousticModel::StateTransitionIndex i = (isSilence ? Am::TransitionModel::silence : Am::TransitionModel::phone0);
    return (*acousticModel_->stateTransition(i))[distance];
}

/*** disabled functions ***/

void SearchInterface::setGrammar(Fsa::ConstAutomatonRef)
{
    warning("setGrammar ist not supported by this decoder");
}

void SearchInterface::getPartialSentence(Search::SearchAlgorithm::Traceback&)
{
    warning("getPartialSentence is not supported by this decoder");
}

Lattice::ConstWordLatticeRef SearchInterface::getCurrentWordLattice() const
{
    warning("getCurrentWordLattice ist not supported by this decoder");
    return Lattice::ConstWordLatticeRef();
}


void SearchInterface::resetStatistics()
{}

void SearchInterface::logStatistics() const
{}

