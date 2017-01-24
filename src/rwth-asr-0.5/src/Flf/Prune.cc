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
#include <Core/Application.hh>
#include <Core/Component.hh>
#include <Core/Parameter.hh>

#include "FlfCore/Basic.hh"
#include "FlfCore/LatticeInternal.hh"
#include "FlfCore/Utility.hh"
#include "Best.hh"
#include "Copy.hh"
#include "FwdBwd.hh"
#include "Prune.hh"
#include "TimeframeConfusionNetwork.hh"

namespace Flf {

    // -------------------------------------------------------------------------
    class FwdBwdPruningLattice : public SlaveLattice {
    private:
	ConstFwdBwdRef fb_;
	Score threshold_;
    public:
	FwdBwdPruningLattice(ConstLatticeRef l, ConstFwdBwdRef fb, Score threshold) :
	    SlaveLattice(l), fb_(fb), threshold_(threshold) {}
	virtual ~FwdBwdPruningLattice() {}

	virtual ConstStateRef getState(Fsa::StateId sid) const {
	    ConstStateRef sr = fsa_->getState(sid);
	    State *sp = new State(sid, sr->tags(), sr->weight());
	    FwdBwd::State::const_iterator itFbScore = fb_->state(sid).begin();
	    for (State::const_iterator a = sr->begin(); a != sr->end(); ++a, ++itFbScore)
		if (itFbScore->score() <= threshold_)
		    *sp->newArc() = *a;
	    return ConstStateRef(sp);
	}

	virtual std::string describe() const {
	    return Core::form("posterior-pruning(%s,%f)", fsa_->describe().c_str(), threshold_);
	}
    };


    ConstLatticeRef pruneByFwdBwdScores(ConstLatticeRef l, ConstFwdBwdRef fb, Score threshold) {
	return ConstLatticeRef(new FwdBwdPruningLattice(l, fb, fb->min() + threshold));
    }


    namespace {
	const Core::ParameterBool paramRelative(
	    "relative",
	    "threshold is interpreted relative to min. arc score",
	    true);
	const Core::ParameterBool paramAsProbability(
	    "as-probability",
	    "threshold given as probability (like the SRI lattice-tool)",
	    false);
	const Core::ParameterFloat paramThreshold(
	    "threshold",
	    "threshold",
	    Core::Type<Score>::max, 0.0, Core::Type<Score>::max);
    } // namespace

    class FwdBwdPruner::Internal : public Core::Component {
    public:
	Core::XmlChannel configurationChannel;
	Core::XmlChannel statisticsChannel;
	FwdBwdBuilderRef fbBuilder;
	bool isRelative;
	bool asProbability;
	Score t;
    public:
	Internal(const Core::Configuration &config, FwdBwdBuilderRef fbBuilder) :
	    Core::Component(config),
	    configurationChannel(config, "configuration"),
	    statisticsChannel(config, "statistics") {
	    this->fbBuilder = (fbBuilder) ? fbBuilder : FwdBwdBuilder::create(select("fb"));
	    isRelative = paramRelative(config);
	    asProbability = paramAsProbability(config);
	    t = paramThreshold(config);
	    if (t < 0.0)
		warning("Fwd/Bwd pruning threshold is negative; set to 0.");
	    if (t != Core::Type<Score>::max) {
		if (asProbability) {
		    if (t > 1.0)
			criticalError("Probability threshold %f is not in [0.0,1.0].", t);
		    t = (t == 0.0) ? Core::Type<Score>::max : -::log(t);
		}
	    }
	    if (configurationChannel) {
		configurationChannel << Core::XmlOpen("configuration")
		    + Core::XmlAttribute("component", this->name());
		{
		    std::ostream &os(configurationChannel);
		    if (t == 0.0)
			os << "threshold = 0.0; single best" << std::endl;
		    else if (t == Core::Type<Score>::max)
			os << "threshold = inf(p=1.0); no pruning" << std::endl;
		    else
			os << "threshold = " << t << " (p = " << ::exp(-t) << ")" << std::endl;
		    os << "thresholds is " << (isRelative ? "relative to min. fwd/bwd. score" : "absolute") << "." << std::endl;
		}
		configurationChannel << Core::XmlClose("configuration");
	    }
	}
	virtual ~Internal() {}

	Score threshold(ConstFwdBwdRef fb) {
	    Score threshold = (t == Core::Type<Score>::max) ?
		Core::Type<Score>::max :
		(isRelative) ? fb->min() + t : t;
	    if (statisticsChannel) {
		statisticsChannel << Core::XmlOpen("statistics")
		    + Core::XmlAttribute("component", this->name());
		statisticsChannel << Core::XmlFull("threshold", threshold);
		statisticsChannel << Core::XmlClose("statistics");
	    }
	    return threshold;
	}
    };

    FwdBwdPruner::FwdBwdPruner() :internal_(0)  {}

    FwdBwdPruner::~FwdBwdPruner() {
	delete internal_;
    }

    ConstLatticeRef FwdBwdPruner::prune(ConstLatticeRef l, bool trim) {
	std::pair<ConstLatticeRef, ConstFwdBwdRef> fbResult = internal_->fbBuilder->build(l);
	return prune(fbResult.first, fbResult.second, trim);
    }

    ConstLatticeRef FwdBwdPruner::prune(ConstLatticeRef l, ConstFwdBwdRef fb, bool trim) {
	verify(fb);
	if (!l && (l->initialStateId() == Fsa::InvalidStateId))
	    return ConstLatticeRef();
	if (internal_->t == Core::Type<Score>::max)
	    return l;
	if (internal_->t == 0.0)
	    return bestProjection(l).first;
	ConstLatticeRef p = ConstLatticeRef(new FwdBwdPruningLattice(l, fb, internal_->threshold(fb)));
	if (trim) {
	    StaticLatticeRef s = StaticLatticeRef(new StaticLattice);
	    persistent(p, s.get(), 0);
	    trimInPlace(s);
	    if (s && (s->initialStateId() != Fsa::InvalidStateId)) {
		s->setBoundaries(p->getBoundaries());
		p = ConstLatticeRef(s);
	    } else
		p = bestProjection(l).first;
	}
	return p;
    }

    FwdBwdPrunerRef FwdBwdPruner::create(const Core::Configuration &config, FwdBwdBuilderRef fbBuilder) {
	FwdBwdPruner *fwdBwdPruner = new FwdBwdPruner;
	fwdBwdPruner->internal_ = new FwdBwdPruner::Internal(config, fbBuilder);
	return FwdBwdPrunerRef(fwdBwdPruner);
    }


    class FwdBwdPruningNode : public FilterNode {
	typedef Node Precursor;
    public:
	static const Core::ParameterBool paramTrim;
    private:
	FwdBwdPrunerRef fbPruner_;
	bool trim_;
    protected:
	virtual ConstLatticeRef filter(ConstLatticeRef l) {
	    if (!l)
		return ConstLatticeRef();
	    return fbPruner_->prune(l, trim_);
	}
    public:
	FwdBwdPruningNode(const std::string &name, const Core::Configuration &config) :
	    FilterNode(name, config) {}
	virtual ~FwdBwdPruningNode() {}
	virtual void init(const std::vector<std::string> &arguments) {
	    fbPruner_ = FwdBwdPruner::create(config);
	    Core::Component::Message msg = log();
	    trim_ = paramTrim(config);
	    if (trim_)
		msg << "Trim pruned lattice(s).\n";
	}
    };
    const Core::ParameterBool FwdBwdPruningNode::paramTrim(
	"trim",
	"trim after applying thresholding",
	true);
    NodeRef createFwdBwdPruningNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new FwdBwdPruningNode(name, config));
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    namespace {
	struct CnProbabilityWeakOrder {
	    ScoreId posteriorId;
	    CnProbabilityWeakOrder(ScoreId posteriorId) : posteriorId(posteriorId) {}
	    bool operator() (const ConfusionNetwork::Arc &a1, const ConfusionNetwork::Arc &a2) const {
		return a1.scores->get(posteriorId) > a2.scores->get(posteriorId);
	    }
	};
    } //namespace
    void prune(ConstConfusionNetworkRef cnRef, Score threshold, u32 maxSlotSize, bool normalize) {
	if (!cnRef)
	    return;
	if (!cnRef->isNormalized())
	    Core::Application::us()->criticalError("Confusion network pruning does only work for normalized CNs.");
	ConfusionNetwork &cn = const_cast<ConfusionNetwork&>(*cnRef);
	ScoreId posteriorId = cn.normalizedProperties->posteriorId;
	verify(posteriorId != Semiring::InvalidId);
	for (ConfusionNetwork::iterator itSlot = cn.begin(), endSlot = cn.end(); itSlot != endSlot; ++itSlot) {
	    ConfusionNetwork::Slot &slot = *itSlot;
	    std::sort(slot.begin(), slot.end(), CnProbabilityWeakOrder(posteriorId));
	    ConfusionNetwork::Slot::iterator itArc = slot.begin(), endArc = slot.end();
	    Score sum = 0.0;
	    for (u32 i = 0, max = std::min(maxSlotSize, u32(slot.size()));
		 (i < max) && (sum < threshold); ++i, ++itArc)
		sum += itArc->scores->get(posteriorId);
	    if (itArc != endArc) {
		slot.erase(itArc, endArc);
		verify(!slot.empty());
		if (normalize) {
		    Score norm = 1.0 / sum;
		    for (itArc = slot.begin(), endArc = slot.end(); itArc != endArc; ++itArc)
			itArc->scores->multiply(posteriorId, norm);
		}
	    }
	    std::sort(slot.begin(), slot.end());
	}
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    namespace {
	struct PosteriorCnProbabilityWeakOrder {
	    bool operator() (const PosteriorCn::Arc &a1, const PosteriorCn::Arc &a2) const {
		return a1.score > a2.score;
	    }
	};
    } //namespace
    void prune(ConstPosteriorCnRef cnRef, Score threshold, u32 maxSlotSize, bool normalize) {
	if (!cnRef)
	    return;
	PosteriorCn &cn = const_cast<PosteriorCn&>(*cnRef);
	for (PosteriorCn::iterator itSlot = cn.begin(), endSlot = cn.end(); itSlot != endSlot; ++itSlot) {
	    PosteriorCn::Slot &slot = *itSlot;
	    std::sort(slot.begin(), slot.end(), PosteriorCnProbabilityWeakOrder());
	    PosteriorCn::Slot::iterator itArc = slot.begin(), endArc = slot.end();
	    Score sum = 0.0;
	    for (u32 i = 0, max = std::min(maxSlotSize, u32(slot.size()));
		 (i < max) && (sum < threshold); ++i, ++itArc)
		sum += itArc->score;
	    if (itArc != endArc) {
		slot.erase(itArc, endArc);
		verify(!slot.empty());
		if (normalize) {
		    Score norm = 1.0 / sum;
		    for (itArc = slot.begin(), endArc = slot.end(); itArc != endArc; ++itArc)
			itArc->score *= norm;
		}
	    }
	    std::sort(slot.begin(), slot.end());
	}
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    void removeEpsSlots(ConstConfusionNetworkRef cnRef, Score threshold) {
	if (!cnRef)
	    return;
	ConfusionNetwork &cn = const_cast<ConfusionNetwork&>(*cnRef);
	ConfusionNetwork::iterator itFrom = cn.begin(), itTo = cn.begin(), endSlot = cn.end();
	std::vector<Fsa::StateId> slotIdMapping(cn.size(), Fsa::InvalidStateId);
	std::vector<u32>::iterator itTargetSlotId = slotIdMapping.begin();
	u32 targetSlotId = 0;
	if (!cnRef->isNormalized()) {
	    if (threshold < 1.0)
		Core::Application::us()->warning("Epsilon slot removal for non-normalized CNs does not support thresholding.");
	    for (; itFrom != endSlot; ++itFrom, ++itTargetSlotId) {
		const ConfusionNetwork::Slot &from = *itFrom;
		for (ConfusionNetwork::Slot::const_iterator itArc = from.begin(), endArc = from.end(); itArc != endArc; ++itArc)
		    if (itArc->label != Fsa::Epsilon) {
			if (itTo != itFrom)
			    *itTo = from;
			++itTo;
			*itTargetSlotId = targetSlotId++;
			break;
		    }
	    }
	} else {
	    ScoreId posteriorId = (threshold != Core::Type<Score>::max) ?
		cnRef->normalizedProperties->posteriorId : Semiring::InvalidId;
	    for (; itFrom != endSlot; ++itFrom, ++itTargetSlotId) {
		const ConfusionNetwork::Slot &from = *itFrom;
		if ((from.front().label != Fsa::Epsilon)
		    || ((from.size() > 1)
			&& ((posteriorId == Semiring::InvalidId) || (from.front().scores->get(posteriorId) < threshold)))) {
		    if (itTo != itFrom)
			*itTo = from;
		    ++itTo;
		    *itTargetSlotId = targetSlotId++;
		}
	    }
	}
	cn.erase(itTo, cn.end());
	verify(cn.size() == targetSlotId);
	if (cn.hasMap()) {
	    ConfusionNetwork::MapProperties &map = const_cast<ConfusionNetwork::MapProperties&>(*cn.mapProperties);
	    std::vector<u32>::const_iterator itTargetSlotId = slotIdMapping.begin();
	    for (Core::Vector<Fsa::StateId>::iterator itSlotIndex = map.slotIndex.begin(), endSlotIndex = map.slotIndex.end();
		 itSlotIndex != endSlotIndex; ++itSlotIndex, ++itTargetSlotId)
		if (*itTargetSlotId != Fsa::InvalidStateId)
		    map.slotIndex[*itTargetSlotId] = *itSlotIndex;
	    map.slotIndex.erase(map.slotIndex.begin() + cn.size(), map.slotIndex.end());
	    for (ConfusionNetwork::MapProperties::Map::iterator itLat2Cn = map.lat2cn.begin(), endLat2Cn = map.lat2cn.end();
		 itLat2Cn != endLat2Cn; ++itLat2Cn) if (itLat2Cn->sid != Fsa::InvalidStateId) {
		    Fsa::StateId cnSid = slotIdMapping[itLat2Cn->sid];
		    if (cnSid == Fsa::InvalidStateId)
			itLat2Cn->sid = itLat2Cn->aid = Fsa::InvalidStateId;
		    else
			itLat2Cn->sid = cnSid;
		}
	}
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    void removeEpsSlots(ConstPosteriorCnRef cnRef, Score threshold) {
	if (!cnRef)
	    return;
	PosteriorCn &cn = const_cast<PosteriorCn&>(*cnRef);
	PosteriorCn::iterator itTo = cn.begin();
	for (PosteriorCn::iterator itFrom = cn.begin(), endSlot = cn.end(); itFrom != endSlot; ++itFrom) {
	    const PosteriorCn::Slot &from = *itFrom;
	    if ((from.front().label != Fsa::Epsilon)
		|| ((from.size() > 1)
		    && (from.front().score < threshold))) {
		if (itTo != itFrom)
		    *itTo = from;
		++itTo;
	    }
	}
	cn.erase(itTo, cn.end());
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    class CnPruningNode : public Node {
	typedef Node Precursor;
    public:
	static const Core::ParameterFloat paramThreshold;
	static const Core::ParameterInt paramMaxSlotSize;
	static const Core::ParameterBool paramNormalize;
	static const Core::ParameterBool paramRemoveEpsSlots;
    protected:
	bool prune_;
	Score threshold_;
	u32 maxSlotSize_;
	bool normalize_;
	bool rmEpsSlots_;
	Score epsSlotThreshold_;
    public:
	CnPruningNode(const std::string &name, const Core::Configuration &config) :
	    Node(name, config) {}
	virtual ~CnPruningNode() {}
	virtual void init(const std::vector<std::string> &arguments) {
	    threshold_ = paramThreshold(config);
	    verify(threshold_ > 0.0);
	    maxSlotSize_ = paramMaxSlotSize(config);
	    verify(maxSlotSize_ > 0);
	    normalize_ = paramNormalize(config);
	    prune_ = (threshold_ != Core::Type<Score>::max) || (maxSlotSize_ != Core::Type<u32>::max);
	    rmEpsSlots_ = paramRemoveEpsSlots(config);
	    epsSlotThreshold_ = paramThreshold(select("eps-slot-removal"));
	    Core::Component::Message msg = log();
	    if (prune_) {
		msg << "Prune";
		if (threshold_ != Core::Type<Score>::max)
		    msg << ", threshold = " << threshold_;
		if (maxSlotSize_ != Core::Type<u32>::max)
		    msg << ", max. slot size = " << maxSlotSize_;
		msg << "\n";
		if (normalize_)
		    msg << "Re-normalize slot-wise posterior prob. dist. after pruning.\n";
	    }
	    if (rmEpsSlots_) {
		msg << "Remove epsilon slots";
		if (epsSlotThreshold_ != Core::Type<Score>::max)
		    msg << ", threshold = " << epsSlotThreshold_;
		msg << "\n";
	    }
	}
    };
    const Core::ParameterFloat CnPruningNode::paramThreshold(
	"threshold",
	"probability threshold",
	Core::Type<Score>::max);
    const Core::ParameterInt CnPruningNode::paramMaxSlotSize(
	"max-slot-size",
	"max. slot size",
	Core::Type<u32>::max);
    const Core::ParameterBool CnPruningNode::paramNormalize(
	"normalize",
	"normalize",
	true);
    const Core::ParameterBool CnPruningNode::paramRemoveEpsSlots(
	"remove-eps-slots",
	"remove eps slots",
	false);

    class NormalizedCnPruningNode : public CnPruningNode {
	typedef CnPruningNode Precursor;
    public:
	NormalizedCnPruningNode(const std::string &name, const Core::Configuration &config) :
	    Precursor(name, config) {}
	virtual ~NormalizedCnPruningNode() {}

	virtual ConstConfusionNetworkRef sendCn(Port to) {
	    verify(connected(to));
	    ConstConfusionNetworkRef cn = requestCn(to);
	    if (cn) {
		if (prune_)
		    prune(cn, threshold_, maxSlotSize_, normalize_);
		if (rmEpsSlots_)
		    removeEpsSlots(cn, epsSlotThreshold_);
	    }
	    return cn;
	}
    };
    NodeRef createNormalizedCnPruningNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new NormalizedCnPruningNode(name, config));
    }

    class PosteriorCnPruningNode : public CnPruningNode {
	typedef CnPruningNode Precursor;
    public:
	PosteriorCnPruningNode(const std::string &name, const Core::Configuration &config) :
	    Precursor(name, config) {}
	virtual ~PosteriorCnPruningNode() {}

	virtual ConstPosteriorCnRef sendPosteriorCn(Port to) {
	    verify(connected(to));
	    ConstPosteriorCnRef cn = requestPosteriorCn(to);
	    if (cn) {
		if (prune_)
		    prune(cn, threshold_, maxSlotSize_, normalize_);
		if (rmEpsSlots_)
		    removeEpsSlots(cn, epsSlotThreshold_);
	    }
	    return cn;
	}
    };
    NodeRef createPosteriorCnPruningNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new PosteriorCnPruningNode(name, config));
    }
    // -------------------------------------------------------------------------

} // namespace Flf
