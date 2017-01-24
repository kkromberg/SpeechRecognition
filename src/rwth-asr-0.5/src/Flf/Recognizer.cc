// Copyright 2011 RWTH Aachen University. All rights reserved.
//
// Licensed under the RWTH ASR License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <Fsa/Cache.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Rational.hh>
#include <Fsa/Static.hh>
#include <Fsa/Sort.hh>
#include <Lattice/Lattice.hh>
#include <Lm/Module.hh>
#include <Lm/FsaLm.hh>
#include <Mm/Feature.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Recognizer.hh>
#include <Search/Search.hh>

#include "FlfCore/Basic.hh"
#include "FlfCore/Utility.hh"
#include "Best.hh"
#include "Convert.hh"
#include "Copy.hh"
#include "EpsilonRemoval.hh"
#include "FwdBwd.hh"
#include "Info.hh"
#include "Lexicon.hh"
#include "Map.hh"
#include "NonWordFilter.hh"
#include "Prune.hh"
#include "SegmentwiseSpeechProcessor.hh"
#include "TimeframeConfusionNetworkBuilder.hh"


namespace Flf {

    // -------------------------------------------------------------------------
    class Recognizer : public Speech::Recognizer, public SegmentwiseSpeechProcessor {
    public:
	static const Core::ParameterFloat paramPartialResultInterval;
	static const Core::ParameterBool paramPronunciationScore;
	static const Core::ParameterBool paramConfidenceScore;
	static const Core::ParameterFloat paramAlpha;
	static const Core::ParameterBool paramApplyNonWordClosureFilter;
	static const Core::ParameterBool paramApplyPosteriorPruning;
	static const Core::ParameterFloat paramThreshold;

    private:
	ModelCombinationRef mc_;
	Core::XmlChannel tracebackChannel_;
	Search::SearchAlgorithm::Traceback traceback_;
	std::vector<Flow::Timestamp> featureTimes_;
	Flow::Time partialResultInterval_;
	Flow::Time lastPartialResult_;

	bool addPronunciationScores_;
	bool addConfidenceScores_;
	bool applyNonWordClosureFilter_;
	bool applyPosteriorPruning_;

	Core::Ref<const Bliss::LemmaPronunciationAlphabet> lpAlphabet_;
	Fsa::LabelId sentenceEndLabel_;
	LabelMapRef nonWordToEpsilonMap_;
	Score pronScale_, lmScale_;
	ScoreId amId_, pronunciationId_, lmId_, confidenceId_;
	ConstSemiringRef semiring_, posteriorSemiring_;
	Score fwdBwdThreshold_;

	const Bliss::SpeechSegment *segment_;

    protected:
	void enterSpeechSegment(const Bliss::SpeechSegment *segment) {
	    if (!segment->orth().empty()) {
		clog() << Core::XmlOpen("orth") + Core::XmlAttribute("source", "reference")
		       << segment->orth()
		       << Core::XmlClose("orth");
	    }
	    recognizer_->resetStatistics();
	    recognizer_->restart();
	    traceback_.clear();

	    acousticModel_->setKey(segment->fullName());

	}

	void leaveSpeechSegment(const Bliss::SpeechSegment *segment) {
	    recognizer_->logStatistics();
	}

	void addPartialToTraceback(Search::SearchAlgorithm::Traceback &partialTraceback) {
	    if(!traceback_.empty()
	       && traceback_.back().time == partialTraceback.front().time)
		partialTraceback.erase(partialTraceback.begin());
	    traceback_.insert(traceback_.end(), partialTraceback.begin(), partialTraceback.end());
	}

	virtual void process(const FeatureList &features) {
	    Core::Ref<const Mm::ScaledFeatureScorer> scorer = acousticModel_->featureScorer();
	    for (FeatureList::const_iterator itFeature = features.begin(), endFeature = features.end();
		 itFeature != endFeature; ++itFeature) {
		FeatureRef f = *itFeature;
		if (scorer->isBuffered() && !scorer->bufferFilled())
		    scorer->addFeature(f);
		else
		    recognizer_->feed(scorer->getScorer(f));
		if (tracebackChannel_.isOpen())
		    featureTimes_.push_back(f->timestamp());
		if (partialResultInterval_ < Core::Type<f32>::max) {
		    if (f->timestamp().startTime() - lastPartialResult_ > partialResultInterval_) {
			Search::SearchAlgorithm::Traceback partialTraceback;
			recognizer_->getPartialSentence(partialTraceback);
			addPartialToTraceback(partialTraceback);
			lastPartialResult_ = f->timestamp().startTime();
		    }
		}
	    }
	    if (scorer->isBuffered()) {
		while (!scorer->bufferEmpty())
		    recognizer_->feed(scorer->flush());
	    }
	}

	void processResult() {
	    Search::SearchAlgorithm::Traceback remainingTraceback;
	    recognizer_->getCurrentBestSentence(remainingTraceback);
	    addPartialToTraceback(remainingTraceback);

	    Core::XmlWriter &os(clog());
	    os << Core::XmlOpen("traceback");
	    traceback_.write(os, lexicon_->phonemeInventory());
	    os << Core::XmlClose("traceback");
	    os << Core::XmlOpen("orth") + Core::XmlAttribute("source", "recognized");
	    for (u32 i = 0; i < traceback_.size(); ++i)
		if (traceback_[i].pronunciation)
		    os << traceback_[i].pronunciation->lemma()->preferredOrthographicForm()
		       << Core::XmlBlank();
	    os << Core::XmlClose("orth");
	    if (tracebackChannel_.isOpen()) {
		logTraceback(traceback_);
		featureTimes_.clear();
	    }
	}

	/*
	  In case of a valid label id:
	  Before:
	  am_score = emission_scale * emission + transition_scale * transition
	  am_scale = 1.0
	  lm_score = pronunciation_scale * pronunciation + lm_scale * lm
	  lm_scale = 1.0
	  Afterwards:
	  am_score = emission_scale * emission + transition_scale * transition + pronunciation_scale * pronunciation
	  am_scale = 1.0
	  lm_score = lm
	  lm_scale = lm_scale
	*/
	ScoresRef buildScore(Fsa::LabelId label, Score amRecogScore, Score lmRecogScore) {
	    verify(amRecogScore != Semiring::Zero);
	    verify(lmRecogScore != Semiring::Zero);

	    Score amScore, pronScore, lmScore;
	    if ((Fsa::FirstLabelId <= label) && (label <= Fsa::LastLabelId)) {
		amScore = amRecogScore;
		pronScore = lpAlphabet_->lemmaPronunciation(label)->pronunciationScore();
		verify(pronScore != Semiring::Zero);
		lmScore = (lmRecogScore - pronScale_ * pronScore) / lmScale_;
	    } else {
		amScore = amRecogScore;
		pronScore = Semiring::One;
		lmScore = lmRecogScore / lmScale_;
	    }

	    ScoresRef scores = semiring_->create();
	    if (addPronunciationScores_) {
		scores->set(amId_, amScore);
		scores->set(pronunciationId_, pronScore);
	    } else
		scores->set(amId_, amScore + pronScale_ * pronScore);
	    scores->set(lmId_, lmScore);
	    if (addConfidenceScores_)
		scores->set(confidenceId_, 0.0);
	    verify(semiring_->project(scores) != Semiring::Zero);

	    return scores;
	}

	ConstLatticeRef buildLattice() {
	    Core::Ref<const ::Lattice::WordLattice> lattice = recognizer_->getCurrentWordLattice();
	    Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
	    Fsa::ConstAutomatonRef amFsa = lattice->part(::Lattice::WordLattice::acousticFsa);
	    Fsa::ConstAutomatonRef lmFsa = lattice->part(::Lattice::WordLattice::lmFsa);
	    require_(Fsa::isAcyclic(amFsa) && Fsa::isAcyclic(lmFsa));

	    StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
	    StaticLatticeRef s = StaticLatticeRef(new StaticLattice);
	    s->setType(Fsa::TypeAcceptor);
	    s->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
	    s->setInputAlphabet(lpAlphabet_);
	    s->setSemiring(semiring_);
	    s->setDescription(Core::form("recog(%s)", segment_->name().c_str()));
	    s->setBoundaries(ConstBoundariesRef(b));
	    s->setInitialStateId(0);

	    Fsa::Stack<Fsa::StateId> S;
	    Core::Vector<Fsa::StateId> sidMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
	    sidMap[amFsa->initialStateId()] = 0;
	    S.push_back(amFsa->initialStateId());
	    Fsa::StateId nextSid = 2;
	    Time finalTime = 0;
	    while (!S.isEmpty()) {
		Fsa::StateId sid = S.pop();
		verify(sid < sidMap.size());
		const ::Lattice::WordBoundary &boundary((*boundaries)[sid]);
		Fsa::ConstStateRef amSr = amFsa->getState(sid);
		Fsa::ConstStateRef lmSr = lmFsa->getState(sid);
		State *sp = new State(sidMap[sid]);
		s->setState(sp);
		b->set(sp->id(), Boundary(
			   boundary.time(),
			   Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
		if (amSr->isFinal()) {
		    sp->newArc(1, buildScore(Fsa::InvalidLabelId, amSr->weight(), lmSr->weight()), sentenceEndLabel_);
		    finalTime = std::max(finalTime, boundary.time());
		}
		for (Fsa::State::const_iterator am_a = amSr->begin(), lm_a = lmSr->begin(); (am_a != amSr->end()) && (lm_a != lmSr->end()); ++am_a, ++lm_a) {
		    sidMap.grow(am_a->target(), Fsa::InvalidStateId);
		    if (sidMap[am_a->target()] == Fsa::InvalidStateId) {
			sidMap[am_a->target()] = nextSid++;
			S.push(am_a->target());
		    }
		    Fsa::ConstStateRef targetAmSr = amFsa->getState(am_a->target());
		    Fsa::ConstStateRef targetLmSr = amFsa->getState(lm_a->target());
		    if (targetAmSr->isFinal() && targetLmSr->isFinal()) {
			if (am_a->input() == Fsa::Epsilon) {
			    ScoresRef scores = buildScore(am_a->input(), am_a->weight(), lm_a->weight());
			    scores->add(amId_, Score(targetAmSr->weight()));
			    scores->add(lmId_, Score(targetLmSr->weight()) / lmScale_);
			    sp->newArc(1, scores, sentenceEndLabel_);
			} else
			    sp->newArc(sidMap[am_a->target()], buildScore(am_a->input(), am_a->weight(), lm_a->weight()), am_a->input());
		    } else
			sp->newArc(sidMap[am_a->target()], buildScore(am_a->input(), am_a->weight(), lm_a->weight()), am_a->input());
		}
	    }
	    State *sp = new State(1);
	    sp->setFinal(semiring_->clone(semiring_->one()));
	    s->setState(sp);
	    b->set(sp->id(), Boundary(finalTime));
	    //s = trimInPlace(s);
	    ConstLatticeRef l = ConstLatticeRef(s);
	    if (applyNonWordClosureFilter_) {
		l = transducer(l);
		l = applyOneToOneLabelMap(l, nonWordToEpsilonMap_);
		StaticLatticeRef filteredLattice = applyEpsClosureWeakDeterminizationFilter(l);
		trimInPlace(filteredLattice);
		l = projectOutput(filteredLattice);
	    }
	    if (addConfidenceScores_ || applyPosteriorPruning_) {
		std::pair<ConstLatticeRef, ConstFwdBwdRef> latAndFb = FwdBwd::build(l, posteriorSemiring_);
		l = latAndFb.first; ConstFwdBwdRef fb = latAndFb.second;
		if (addConfidenceScores_) {
		    ConstPosteriorCnRef cn = buildFramePosteriorCn(l, fb);
		    l = extendByFCnConfidence(l, cn, confidenceId_, RescoreModeInPlaceCache);
		    l = persistent(l);
		}
		if (applyPosteriorPruning_) {
		    l = pruneByFwdBwdScores(l, fb, fwdBwdThreshold_);
		    StaticLatticeRef trimmedLattice = StaticLatticeRef(new StaticLattice);
		    copy(l, trimmedLattice.get(), 0);
		    trimInPlace(trimmedLattice);
		    trimmedLattice->setBoundaries(l->getBoundaries());
		    l = normalizeCopy(trimmedLattice);
		}
	    }
	    return l;
	}

	void logTraceback(const Search::SearchAlgorithm::Traceback &traceback) {
	    tracebackChannel_ << Core::XmlOpen("traceback") + Core::XmlAttribute("type", "xml");
	    u32 previousIndex = traceback.begin()->time;
	    Search::SearchAlgorithm::ScoreVector previousScore(0.0, 0.0);
	    for (std::vector<Search::SearchAlgorithm::TracebackItem>::const_iterator tbi = traceback.begin(); tbi != traceback.end(); ++tbi) {
		if (tbi->pronunciation) {
		    tracebackChannel_ << Core::XmlOpen("item") + Core::XmlAttribute("type", "pronunciation")
				      << Core::XmlFull("orth", tbi->pronunciation->lemma()->preferredOrthographicForm())
				      << Core::XmlFull("phon", tbi->pronunciation->pronunciation()->format(lexicon_->phonemeInventory()))
				      << Core::XmlFull("score", f32(tbi->score.acoustic - previousScore.acoustic))
			+ Core::XmlAttribute("type", "acoustic")
				      << Core::XmlFull("score", f32(tbi->score.lm - previousScore.lm))
			+ Core::XmlAttribute("type", "language");
		    if(previousIndex < tbi->time)
			tracebackChannel_ << Core::XmlEmpty("samples")
			    + Core::XmlAttribute("start", f32(featureTimes_[previousIndex].startTime())) +
			    Core::XmlAttribute("end", f32(featureTimes_[tbi->time - 1].endTime()))
					  << Core::XmlEmpty("features") + Core::XmlAttribute("start", previousIndex) +
			    Core::XmlAttribute("end", tbi->time - 1);
		    tracebackChannel_ << Core::XmlClose("item");
		}
		previousScore = tbi->score;
		previousIndex = tbi->time;
	    }
	    tracebackChannel_ << Core::XmlClose("traceback");
	}

    public:
	Recognizer(const Core::Configuration &config, ModelCombinationRef mc):
	    Core::Component(config),
	    Speech::Recognizer(config),
	    SegmentwiseSpeechProcessor(config, mc),
	    mc_(mc),
	    tracebackChannel_(config, "traceback") {
	    require(mc_);
	    partialResultInterval_ = paramPartialResultInterval(config);
	    lastPartialResult_ = 0;
	    pronScale_ = mc_->pronunciationScale();
	    lmScale_ = mc_->languageModel()->scale();
	    addPronunciationScores_ = paramPronunciationScore(config);
	    addConfidenceScores_ = paramConfidenceScore(config);
	    applyNonWordClosureFilter_ = paramApplyNonWordClosureFilter(config);
	    applyPosteriorPruning_ = paramApplyPosteriorPruning(config);
	    {
		Core::Component::Message msg(log());
		lpAlphabet_ = mc_->lexicon()->lemmaPronunciationAlphabet();
		sentenceEndLabel_ = Fsa::Epsilon;
		const Bliss::Lemma *special = mc_->lexicon()->specialLemma("sentence-end");
		if (special) {
		    Bliss::Lemma::LemmaPronunciationRange lpRange = special->pronunciations();
		    if (lpRange.first != lpRange.second) sentenceEndLabel_ = lpRange.first->id();
		}
		msg << "Sentence end symbol is \"" << lpAlphabet_->symbol(sentenceEndLabel_) << "\".\n";

		u32 dim = 0;
		amId_ = dim++;
		lmId_ = dim++;
		if (addPronunciationScores_)
		    pronunciationId_ = dim++;
		else
		    pronunciationId_ = Semiring::InvalidId;
		if (addConfidenceScores_)
		    confidenceId_ = dim++;
		else
		    confidenceId_ = Semiring::InvalidId;
		semiring_ = Semiring::create(Fsa::SemiringTypeTropical, dim);
		semiring_->setKey(amId_, "am");
		semiring_->setScale(amId_, 1.0);
		semiring_->setKey(lmId_, "lm");
		semiring_->setScale(lmId_, lmScale_);
		if (addPronunciationScores_) {
		    semiring_->setKey(pronunciationId_, "pronunciation");
		    semiring_->setScale(pronunciationId_, pronScale_);
		}
		if (addConfidenceScores_) {
		    semiring_->setKey(confidenceId_, "confidence");
		    semiring_->setScale(confidenceId_, 0.0);
		}
		msg << "Semiring is " << semiring_->name() << ".\n";
		if (addConfidenceScores_ || applyPosteriorPruning_) {
		    posteriorSemiring_ = toLogSemiring(semiring_, paramAlpha(select("fb")));
		    msg << "Posterior-semiring is " << posteriorSemiring_->name() << ".\n";
		}
		if (applyNonWordClosureFilter_) {
		    nonWordToEpsilonMap_ = LabelMap::createNonWordToEpsilonMap(Lexicon::LemmaPronunciationAlphabetId);
		    msg << "Non-word-closure filter is active.\n";
		}
		if (addConfidenceScores_) {
		    msg << "Confidence score calculation is active (Attention: Confidence scores are calculated on lemma pronunciations). \n";
		}
		if (applyPosteriorPruning_) {
		    fwdBwdThreshold_ = paramThreshold(select("posterior-pruning"));
		    msg << "Posterior pruning is active (threshold=" << fwdBwdThreshold_ << ").\n";
		}
	    }
	    initializeRecognizer(*mc_);
	    verify(recognizer_);
	}

	void setGrammar(Fsa::ConstAutomatonRef fsa) {
	    require(Lexicon::us()->alphabetId(fsa->getInputAlphabet()) == Lexicon::SyntacticTokenAlphabetId);
	    recognizer_->setGrammar(fsa);
	}

	ConstLatticeRef recognize(const Bliss::SpeechSegment *segment) {
	    segment_ = segment;
	    enterSpeechSegment(segment);
	    processSegment(segment);
	    processResult();
	    ConstLatticeRef l = buildLattice();
	    info(l, clog());
	    leaveSpeechSegment(segment);
	    segment_ = 0;
	    return l;
	}
    };
    const Core::ParameterFloat Recognizer::paramPartialResultInterval(
	"partial-result-interval",
	"time between attempt to produce partial recognition results (in seconds)",
	Core::Type<f32>::max, 0.0);
    const Core::ParameterBool Recognizer::paramPronunciationScore(
	"add-pronunication-score",
	"add an extra dimension containing the pronunciation score",
	false);
    const Core::ParameterBool Recognizer::paramApplyNonWordClosureFilter(
	"apply-non-word-closure-filter",
	"apply the non word closure filter",
	false);
    const Core::ParameterBool Recognizer::paramConfidenceScore(
	"add-confidence-score",
	"add an extra dimension containing the confidence score",
	false);
    const Core::ParameterFloat Recognizer::paramAlpha(
	"alpha",
	"scale dimensions for posterior calculation",
	0.0);
    const Core::ParameterBool Recognizer::paramApplyPosteriorPruning(
	"apply-posterior-pruning",
	"prune posterior",
	false);
    const Core::ParameterFloat Recognizer::paramThreshold(
	"threshold",
	"posterior pruning threshold",
	200);

    // -------------------------------------------------------------------------



    // -------------------------------------------------------------------------

    class RecognizerNode : public Node {
    public:
	static const Core::ParameterString paramGrammarKey;
	static const Core::ParameterInt paramGrammarArcsLimit;
	static const Core::ParameterFloat paramGrammarDensity;
	static const Core::ParameterFloat paramGrammarPruningStepWidth;;

    private:
	ModelCombinationRef mc_;
	Recognizer *recognizer_;

	Core::XmlChannel grammarChannel_;
	Key grammarKey_;
	size_t grammarMaxArcs_;
	f32 grammarDensity_;
	f32 grammarPruningStepWidth_;

	ConstLatticeRef l_;
	bool isBuffered_;

    protected:
	ScaledLanguageModelRef buildEmptyFsaGrammarLm() {
	    Fsa::StaticAutomaton *s = new Fsa::StaticAutomaton;
	    s->setType(Fsa::TypeAcceptor);
	    s->setInputAlphabet(Lexicon::us()->syntacticTokenAlphabet());
	    s->setSemiring(Fsa::TropicalSemiring);
	    s->setDescription("empty-grammar");
	    s->setInitialStateId(s->newFinalState(Fsa::TropicalSemiring->one())->id());
	    Fsa::ConstAutomatonRef f = Fsa::ConstAutomatonRef(s);
	    Lm::FsaLm *fsaLm = new Lm::FsaLm(select("lm"), Bliss::LexiconRef(Lexicon::us()));
	    fsaLm->setFsa(f);
	    ScaledLanguageModelRef lm = Lm::Module::instance().createScaledLanguageModel(select("lm"), LanguageModelRef(fsaLm));
	    return lm;
	}

	Fsa::ConstAutomatonRef buildFsaGrammar(ConstLatticeRef l) {
	    if (!l || (l->initialStateId() == Fsa::InvalidStateId))
		return Fsa::ConstAutomatonRef();

	    grammarChannel_ << Core::XmlOpen("grammar");

	    /*
	      the initial grammar
	    */
	    LabelMapRef nonWordToEpsilonMap = LabelMap::createNonWordToEpsilonMap(Lexicon::us()->alphabetId(l->getInputAlphabet()));
	    LatticeCounts bestCounts = count(applyOneToOneLabelMap(best(l), nonWordToEpsilonMap));
	    u32 nBestWords = bestCounts.nArcs_ - bestCounts.nIEps_;
	    LatticeCounts latticeCounts = count(l);
	    f32 latticeDensity = (nBestWords == 0) ? f32(0) : f32(latticeCounts.nArcs_) / f32(nBestWords);
	    grammarChannel_ << Core::XmlOpen("lattice");
	    info(l, grammarChannel_);
	    grammarChannel_ << Core::XmlFull("density", latticeDensity);
	    grammarChannel_ << Core::XmlClose("lattice");

	    /*
	      to many arcs
	      -> apply non-word-closure-determinization-filter
	      -> dynamic pruning
	    */
	    /*
	    if (latticeCounts.nArcs_ > grammarMaxArcs_) {
		l = transducer(projectInput(l));
		l = mapInput(l, MapToLemma);
		l = applyOneToOneLabelMap(l, nonWordToEpsilonMap);
		l = applyEpsClosureDeterminizationFilter(l);
		l = projectOutput(l);
		latticeCounts = count(l);
		density = (nBestWords == 0) ? f32(0) : f32(latticeCounts.nArcs_) / f32(nBestWords);
		grammarChannel_ << Core::XmlOpen("lattice-after-non-word-closure-determinization-filter");
		info(l, grammarChannel_);
		grammarChannel_ << Core::XmlFull("density", density);
		grammarChannel_ << Core::XmlClose("lattice-after-non-word-closure-determinization-filter");
	    }
	    */

	    f32 desiredDensity = std::min(latticeDensity, grammarDensity_);
	    if ((desiredDensity < latticeDensity) || (latticeCounts.nArcs_ > grammarMaxArcs_)) {
		StaticLatticeRef
		    tmpL = StaticLatticeRef(new StaticLattice),
		    prunedL = StaticLatticeRef(new StaticLattice);
		std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult = FwdBwd::build(l, toLogSemiring(l->semiring()));
		l = fbResult.first;
		Score threshold = grammarPruningStepWidth_;
		LatticeCounts prunedCounts;
		prunedCounts.nArcs_ = 0;
		f32 prunedDensity = 0.0;
		for (;;) {
		    persistent(pruneByFwdBwdScores(l, fbResult.second, threshold), tmpL.get(), 0);
		    trimInPlace(tmpL);
		    LatticeCounts tmpCounts;
		    f32 tmpDensity;
		    if (tmpL->initialStateId() == Fsa::InvalidStateId) {
			tmpCounts.nArcs_ = 0;
			tmpDensity = 0.0;
		    } else {
			tmpCounts = count(tmpL);
			tmpDensity = (nBestWords == 0) ? f32(0) : f32(tmpCounts.nArcs_) / f32(nBestWords);
		    }
		    grammarChannel_ << Core::XmlFull(
			"dynamic-pruning", Core::form("threshold = %f, #arcs = %d, density = %f", threshold, s32(tmpCounts.nArcs_), tmpDensity));
		    if (tmpCounts.nArcs_ >= grammarMaxArcs_) {
			if ((prunedCounts.nArcs_ == 0) || ((tmpCounts.nArcs_ - grammarMaxArcs_) < (grammarMaxArcs_ - prunedCounts.nArcs_))) {
			    prunedCounts = tmpCounts;
			    prunedDensity = tmpDensity;
			    prunedL = tmpL;
			}
			break;
		    }
		    prunedCounts = tmpCounts;
		    prunedDensity = tmpDensity;
		    StaticLatticeRef tmp = prunedL; prunedL = tmpL; tmpL = tmp; tmpL->clear();
		    if (tmpDensity >= desiredDensity)
			break;
		    threshold += grammarPruningStepWidth_;
		}
		grammarChannel_ << Core::XmlFull(
		    "dynamic-pruning", Core::form("final threshold = %f, #arcs = %d, density = %f", threshold, s32(prunedCounts.nArcs_), prunedDensity));
		l =prunedL;
		latticeCounts = prunedCounts;
		latticeDensity = prunedDensity;
		grammarChannel_ << Core::XmlOpen("lattice-after-dynamic-pruning");
		info(l, grammarChannel_);
		grammarChannel_ << Core::XmlFull("density", latticeDensity);
		grammarChannel_ << Core::XmlClose("lattice-after-dynamic-pruning");
	    }

	    /*
	      make epsilon-free syntactic token acceptor
	     */
	    l = projectInput(mapInput(l, MapToSyntacticTokenSequence));
	    l = fastRemoveEpsilons(l);
	    l = persistent(l);
	    latticeCounts = count(l);
	    latticeDensity = (nBestWords == 0) ? f32(0) : f32(latticeCounts.nArcs_) / f32(nBestWords);
	    grammarChannel_ << Core::XmlOpen("lattice-after-epsilon-removal");
	    info(l, grammarChannel_);
	    grammarChannel_ << Core::XmlFull("density", latticeDensity);
	    grammarChannel_ << Core::XmlClose("lattice-after-epsilon-removal");

	    /*
	       to fsa, determinization or minimization
	    */
	    Core::Ref<Fsa::StaticAutomaton> g;
	    if (!grammarKey_.empty()) {
		g = Fsa::staticCopy(toFsa(l, l->semiring()->id(grammarKey_)));
	    } else {
		g = Fsa::staticCopy(toFsa(l));
	    }
	    /*
	    // determinize
	    g->addProperties(Fsa::PropertyAcyclic); // just to make sure
	    g = Fsa::staticCopy(Fsa::determinize(g));
	    Fsa::trimInPlace(g);
	    grammarChannel_ << Core::XmlOpen("fsa-after-determinization");
	    Fsa::info(g, grammarChannel_);
	    grammarChannel_ << Core::XmlClose("fsa-after-determinization");
	    */
	    // minimize
	    Core::Ref<Fsa::StaticAutomaton> gT;
	    gT = Fsa::staticCopy(Fsa::transpose(g));
	    gT->addProperties(Fsa::PropertyAcyclic); // just to make sure
	    gT = Fsa::staticCopy(Fsa::determinize(gT));
	    Fsa::trimInPlace(gT);
	    g = Fsa::staticCopy(Fsa::transpose(gT));
	    // grammarChannel_ << Core::XmlOpen("grammar-after-transposed-determinization");
	    // Fsa::info(g, grammarChannel_);
	    // grammarChannel_ << Core::XmlClose("grammar-after-transposed-determinization");
	    g->addProperties(Fsa::PropertyAcyclic); // just to make sure
	    g = Fsa::staticCopy(Fsa::determinize(g));
	    Fsa::trimInPlace(g);
	    grammarChannel_ << Core::XmlOpen("fsa-after-minimization");
	    Fsa::info(g, grammarChannel_);
	    grammarChannel_ << Core::XmlClose("fsa-after-minimization");

	    grammarChannel_ << Core::XmlClose("grammar");

	    if (!g || (g->initialStateId() == Fsa::InvalidStateId))
		return Fsa::ConstAutomatonRef();
	    return g;
	}

    public:
	RecognizerNode(const std::string &name, const Core::Configuration &config) :
	    Node(name, config), mc_(), recognizer_(0), grammarChannel_(Core::Configuration(config, "grammar"), "log") {}
	virtual ~RecognizerNode() {
	    delete recognizer_;
	}

	virtual void init(const std::vector<std::string> &arguments) {
	    if (!connected(1))
		criticalError("Speech segment at port 1 required");
	    if (!Lexicon::us()->isReadOnly())
		warning("Lexicon is not read-only, "
			"dynamically added/modified lemmas are not considered by the recognizer.");
	    AcousticModelRef am = getAm(select("acoustic-model"));
	    ScaledLanguageModelRef lm;
	    if (connected(0)) {
		Core::Component::Message msg(log());
		msg << "Get grammar from port 0.\n";
		grammarKey_ = paramGrammarKey(select("grammar"));
		s32 tmp = paramGrammarArcsLimit(select("grammar"));
		grammarMaxArcs_ = (tmp == Core::Type<s32>::max) ?
		    Core::Type<size_t>::max : size_t(tmp);
		grammarDensity_ = paramGrammarDensity(select("grammar"));
		grammarPruningStepWidth_ = paramGrammarPruningStepWidth(select("grammar"));
		if (!grammarKey_.empty())
		    msg << "Use dimension \"" << grammarKey_ << "\" as grammar.\n";
		lm = buildEmptyFsaGrammarLm();
	    } else {
		lm = getLm(select("lm"));
	    }
	    mc_ = getModelCombination(config, am, lm);
	    recognizer_ = new Recognizer(config, mc_);
	    isBuffered_ = false;
	}

	virtual void finalize() {
	    recognizer_->reset();
	}

	virtual ConstLatticeRef sendLattice(Port to) {
	    if (isBuffered_)
		return l_;
	    const Bliss::SpeechSegment *segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
	    clog() << Core::XmlOpen("layer") + Core::XmlAttribute("name", name);
	    bool skip = false;
	    if (connected(0)) {
		Fsa::ConstAutomatonRef g = buildFsaGrammar(requestLattice(0));
		if (g) {
		    Fsa::info(g, log("Set grammar to:\n"));
		    recognizer_->setGrammar(g);
		} else {
		    warning("Grammar is the empty lattice, skip segment.");
		    skip = true;
		}
	    }
	    if (!skip)
		l_ = recognizer_->recognize(segment);
	    clog() << Core::XmlClose("layer");
	    isBuffered_ = true;
	    return l_;
	}

	virtual void sync() {
	    l_.reset();
	    isBuffered_ = false;
	}
    };
    const Core::ParameterString RecognizerNode::paramGrammarKey(
	"key",
	"dimension storing the grammar scores",
	"");
    const Core::ParameterInt RecognizerNode::paramGrammarArcsLimit(
	"arcs-limit",
	"maximum number of arcs in grammar",
	Core::Type<s32>::max, 0, Core::Type<s32>::max);
    const Core::ParameterFloat RecognizerNode::paramGrammarDensity(
	"density",
	"desired density",
	Core::Type<f32>::max, 1.0, Core::Type<f32>::max);
    const Core::ParameterFloat RecognizerNode::paramGrammarPruningStepWidth(
	"pruning-step",
	"step width between two prunings",
	2.5, 0.0, Core::Type<f32>::max);
    NodeRef createRecognizerNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new RecognizerNode(name, config));
    }

} // namespace Flf

