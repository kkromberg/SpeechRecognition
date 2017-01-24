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
#ifndef _SEARCH_SEARCHALGORITHM_HH
#define _SEARCH_SEARCHALGORITHM_HH

#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <Core/Component.hh>
#include <Lm/LanguageModel.hh>
#include <Lattice/Lattice.hh>
#include <Speech/ModelCombination.hh>
#include "Types.hh"

namespace Search {

    /*
     * Search
     *
     * Abstract interface of search implementations
     */

    class SearchAlgorithm : public virtual Core::Component {
    public:
	typedef Speech::Score Score;
	typedef Speech::TimeframeIndex TimeframeIndex;

	struct ScoreVector {
	    Score acoustic, lm;
	    ScoreVector(Score a, Score l) : acoustic(a), lm(l) {}
	    operator Score() const {
		return acoustic + lm;
	    };
	};
	struct TracebackItem {
	public:
	    typedef Lattice::WordBoundary::Transit Transit;
	public:
	    const Bliss::LemmaPronunciation *pronunciation;
	    TimeframeIndex time;
	    ScoreVector score;
	    Transit transit;
	    TracebackItem(const Bliss::LemmaPronunciation *p, TimeframeIndex t, ScoreVector s, Transit te):
		pronunciation(p), time(t), score(s), transit(te) {}
	};
	class Traceback : public std::vector<TracebackItem> {
	public:
	    void write(std::ostream &os, Core::Ref<const Bliss::PhonemeInventory>) const;
	    Fsa::ConstAutomatonRef lemmaAcceptor(Core::Ref<const Bliss::Lexicon>) const;
	    Fsa::ConstAutomatonRef lemmaPronunciationAcceptor(Core::Ref<const Bliss::Lexicon>) const;
	    Lattice::WordLatticeRef wordLattice(Core::Ref<const Bliss::Lexicon>) const;
	};
    public:
	SearchAlgorithm(const Core::Configuration&);
	virtual ~SearchAlgorithm() {}

	virtual Speech::ModelCombination::Mode modelCombinationNeeded() const;
	virtual bool setModelCombination(const Speech::ModelCombination &modelCombination) = 0;

	virtual void setGrammar(Fsa::ConstAutomatonRef) = 0;

	virtual void restart() = 0;
	virtual void feed(const Mm::FeatureScorer::Scorer&) = 0;
	virtual void getPartialSentence(Traceback &result) = 0;
	virtual void getCurrentBestSentence(Traceback &result) const = 0;
	virtual Lattice::ConstWordLatticeRef getCurrentWordLattice() const = 0;
	virtual void resetStatistics() = 0;
	virtual void logStatistics() const = 0;

	/**
	 * Length of acoustic look-ahead.
	 * 0 means no look-ahead is used.
	 */
	virtual u32 lookAheadLength() {
	    return 0;
	}
	/**
	 * Called before feed() with as many acoustic feature vectors from the future
	 * as returned by lookaheadLength().  At the end of a segment, less than requested,
	 * or even zero feature vectors may be given.
	 */
	virtual void setLookAhead(const std::vector<Mm::FeatureVector>&) {}
    };

} // namespace Search

#endif // _SEARCH_SEARCHALGORITHM_HH
