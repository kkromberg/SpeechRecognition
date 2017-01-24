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
#include "Basic.hh"
#include "Best.hh"
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Project.hh>
#include <Fsa/Prune.hh>
#include <Fsa/RemoveEpsilons.hh>
#include "Basic.hh"
#include "Compose.hh"
#include "Rational.hh"
#include "RemoveEpsilons.hh"
#include "Static.hh"

namespace Lattice {

    /**
     * NBestListExtractor
     */
    NBestListExtractor::NBestListExtractor() :
	targetNHypotheses_(1),
	minThreshold_(100),
	maxThreshold_(Core::Type<f32>::max),
	thresholdIncrement_(10),
	workOnOutput_(true),
	latticeIsDeterministic_(workOnOutput_ ? true : false)
    {}

    NBestListExtractor::~NBestListExtractor()
    {}

    void NBestListExtractor::initialize(Bliss::LexiconRef lexicon)
    {
	require(lexicon);
	lemmaToEval_ =
	    Fsa::multiply(
		lexicon->createLemmaToEvaluationTokenTransducer(),
		Fsa::Weight(0));
	lemmaPronToLemma_ =
	    Fsa::multiply(
		lexicon->createLemmaPronunciationToLemmaTransducer(),
		Fsa::Weight(0));
    }

    Fsa::ConstAutomatonRef NBestListExtractor::mapLemmaPronToEval(
	Fsa::ConstAutomatonRef fsa) const
    {
	return Fsa::cache(
	    Fsa::projectOutput(
		Fsa::composeMatching(
		    Fsa::composeMatching(
			fsa,
			lemmaPronToLemma_),
		    lemmaToEval_)));
    }

    Fsa::ConstAutomatonRef NBestListExtractor::mapEvalToLemmaPron(
	Fsa::ConstAutomatonRef fsa) const
    {
	return Fsa::cache(
	    Fsa::removeEpsilons(
		Fsa::projectInput(
		    Fsa::composeMatching(
			lemmaPronToLemma_,
			Fsa::composeMatching(
			    lemmaToEval_,
			    fsa)))));
    }

    Fsa::ConstAutomatonRef NBestListExtractor::getNBestListWithoutWordBoundaries(
	Fsa::ConstAutomatonRef fsa)
    {
	Fsa::ConstAutomatonRef nbl;
	u32 nHypotheses = 0;
	Fsa::Weight minWeight = Fsa::bestscore(fsa);
	Fsa::ConstSemiringRef sr = fsa->semiring();
	f32 threshold = minThreshold_;
	for (; threshold <= maxThreshold_; threshold += thresholdIncrement_) {
	    //	log("pruning threshold: ") << threshold;

	    Fsa::ConstAutomatonRef pruned = Fsa::prunePosterior(fsa, Fsa::Weight(threshold));
	    nbl = Fsa::nbest(Fsa::cache(pruned), targetNHypotheses_);

	    /**
	     *  Stop if number of hypotheses has not changed since the last iteration.
	     *  It is very likely that the automaton does not contain more hypotheses.
	     */
	    Fsa::ConstStateRef hypotheses = nbl->getState(nbl->initialStateId());
	    bool hasChanged = nHypotheses != hypotheses->nArcs();
	    if (!hasChanged) {
		//	    log("return on \"has-not-changed\"");
		return Fsa::staticCopy(nbl);
	    }

	    nHypotheses = hypotheses->nArcs();

	    /**
	     *  Stop if n-best list is complete. This is the case if the target number
	     *  of hypotheses is reached and the least hypothesis has lower score than
	     *  the relative pruning threshold.
	     */
	    bool hasEnoughHypotheses = nHypotheses == targetNHypotheses_;
	    Fsa::Weight leastWeight = nbl->getState(nbl->initialStateId())->getArc(nHypotheses - 1)->weight();
	    bool hasNBestHypotheses = sr->compare(leastWeight, sr->extend(minWeight, Fsa::Weight(threshold))) < 0;
	    if (hasEnoughHypotheses && hasNBestHypotheses) {
		//	    log("return on \"enough-hypotheses\"");
		return Fsa::staticCopy(nbl);
	    }
	}

	//    log("return on \"maximum-pruning-threshold\"");
	return Fsa::staticCopy(nbl);
    }

    ConstWordLatticeRef NBestListExtractor::getNBestList(
	ConstWordLatticeRef lattice)
    {
	if (targetNHypotheses_ == 1) {
	    return best(lattice, true);
	}

	lattice = removeEpsilons(lattice);
	Fsa::ConstAutomatonRef total = lattice->mainPart();
	require(total->semiring() == Fsa::TropicalSemiring);

	if (!workOnOutput_) {
	    total = mapLemmaPronToEval(Fsa::projectInput(total));
	}
	total = Fsa::projectOutput(total);
	Fsa::ConstAutomatonRef nBestList = getNBestListWithoutWordBoundaries(total);
	Core::Vector<Lattice::ConstWordLatticeRef> hypotheses;
	Fsa::State::const_iterator hIt =
	    nBestList->getState(nBestList->initialStateId())->begin();
	Fsa::State::const_iterator hEnd =
	    nBestList->getState(nBestList->initialStateId())->end();
	for (; hIt != hEnd; ++ hIt) {
	    Fsa::ConstAutomatonRef hf =
		Fsa::partial(
		    nBestList,
		    hIt->target(),
		    hIt->weight());
	    if (!workOnOutput_) {
		hf = mapEvalToLemmaPron(hf);
	    }
	    hf = Fsa::multiply(hf, Fsa::Weight(0));
	    hypotheses.push_back(
		staticCopy(
		    normalize(
			best(
			    composeMatching(
				hf,
				lattice)))));
	}
	return staticCopy(unite(hypotheses));
    }

    ConstWordLatticeRef nbest(ConstWordLatticeRef l, u32 n) {
	NBestListExtractor e;
	e.setNumberOfHypotheses(n);
	return e.getNBestList(l);
    }

    struct SingleBest
    {
	SingleBest() {}
	Fsa::ConstAutomatonRef modify(Fsa::ConstAutomatonRef fsa) {
	    return Fsa::best(fsa);
	}
    };

    ConstWordLatticeRef best(ConstWordLatticeRef lattice, bool listFormat)
    {
	require(lattice->nParts() == 1);
	SingleBest b;
	ConstWordLatticeRef result = apply(lattice, b);
	if (listFormat) {
	    Core::Vector<ConstWordLatticeRef> tmp;
	    tmp.push_back(normalize(result));
	    result = unite(tmp);
	}
	return result;
    }

} // namespace Lattice
