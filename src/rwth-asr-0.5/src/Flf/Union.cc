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
#include <Core/Choice.hh>
#include <Core/Parameter.hh>
#include <Fsa/Hash.hh>

#include "FlfCore/Basic.hh"
#include "FlfCore/LatticeInternal.hh"
#include "Convert.hh"
#include "Union.hh"

namespace Flf {

    // -------------------------------------------------------------------------
    class UnionBoundaries : public Boundaries {
    private:
	Fsa::StateId nLats_;
	ConstBoundariesRefList boundaries_;
	Boundary initialBoundary_;

    private:
	std::pair<Fsa::StateId, Fsa::StateId> fromUnionSid(Fsa::StateId unionSid) const {
	    Fsa::StateId sid = (unionSid - 1) / nLats_;
	    return std::make_pair(unionSid - 1 - sid * nLats_, sid);
	}

    public:
	UnionBoundaries(const ConstLatticeRefList &lats) :
	    Boundaries(), nLats_(lats.size()),
	    boundaries_(lats.size()) {
	    boundaries_[0] = lats[0]->getBoundaries();
	    initialBoundary_ = boundaries_[0]->get(lats[0]->initialStateId());
	    for (u32 i = 1; i < nLats_; ++i) {
		boundaries_[i] = lats[i]->getBoundaries();
		if (initialBoundary_.valid()) {
		    const Boundary &currentInitialBoundary(
			boundaries_[i]->get(lats[i]->initialStateId()));
		    if (currentInitialBoundary.time() != initialBoundary_.time())
			initialBoundary_ = InvalidBoundary;
		    else if (!(currentInitialBoundary.transit() == initialBoundary_.transit()))
			initialBoundary_.setTransit(Boundary::Transit());
		}
	    }
	}
	virtual ~UnionBoundaries() {}

	virtual bool valid() const {
	    return true;
	}

	virtual bool valid(Fsa::StateId unionSid) const {
	    return get(unionSid).valid();
	}

	virtual const Boundary& get(Fsa::StateId unionSid) const {
	    if (unionSid == 0)
		return initialBoundary_;
	    else {
		std::pair<Fsa::StateId, Fsa::StateId> sid = fromUnionSid(unionSid);
		return boundaries_[sid.first]->get(sid.second);
	    }
	}
    };


    class UnionLattice : public SlaveLattice {
    private:
	Fsa::StateId nLats_;
	ConstLatticeRefList lats_;
	ConstSemiringRef semiring_;

    private:
	Fsa::StateId toUnionSid(Fsa::StateId i, Fsa::StateId sid) const {
	    return sid * nLats_ + i + 1;
	}

	std::pair<Fsa::StateId, Fsa::StateId> fromUnionSid(Fsa::StateId unionSid) const {
	    Fsa::StateId sid = (unionSid - 1) / nLats_;
	    return std::make_pair(unionSid - 1 - sid * nLats_, sid);
	}

    public:
	UnionLattice(const ConstLatticeRefList &lats, ConstSemiringRef semiring) :
	    SlaveLattice(lats[0]), nLats_(lats.size()),
	    lats_(lats), semiring_(semiring) {
	    verify(semiring);
	    bool hasBoundaries = false;
	    for (u32 i = 0; i < nLats_; ++i) {
		ConstLatticeRef currentLattice = lats[i];
		verify(fsa_->type() == currentLattice->type());
		verify(fsa_->getInputAlphabet().get() == currentLattice->getInputAlphabet().get());
		if (fsa_->type() != Fsa::TypeAcceptor)
		    verify(fsa_->getOutputAlphabet().get() ==
			   currentLattice->getOutputAlphabet().get());
		if (lats[i]->getBoundaries()->valid())
		    hasBoundaries = true;
	    }
	    if (hasBoundaries)
		setBoundaries(ConstBoundariesRef(new UnionBoundaries(lats)));
	}
	virtual ~UnionLattice() {}

	virtual ConstSemiringRef semiring() const {
	    return semiring_;
	}

	virtual Fsa::StateId initialStateId() const {
	    return 0;
	}

	virtual ConstStateRef getState(Fsa::StateId unionSid) const {
	    State *sp;
	    if (unionSid == 0) {
		sp = new State(0);
		for (u32 i = 0; i < nLats_; ++i) {
		    ConstStateRef sr = lats_[i]->getState(lats_[i]->initialStateId());
		    if (sr->isFinal())
			sp->newArc(toUnionSid(i, lats_[i]->initialStateId()),
				   semiring_->one(), Fsa::Epsilon);
		    for (State::const_iterator a = sr->begin(); a != sr->end(); ++a)
			sp->newArc(toUnionSid(i, a->target()), a->weight(),
				   a->input(), a->output());
		}
	    } else {
		std::pair<Fsa::StateId, Fsa::StateId> sid = fromUnionSid(unionSid);
		ConstStateRef sr = lats_[sid.first]->getState(sid.second);
		sp = new State(*sr);
		sp->setId(unionSid);
		for (State::iterator a = sp->begin(); a != sp->end(); ++a)
		    a->target_ = toUnionSid(sid.first, a->target());
	    }
	    return ConstStateRef(sp);
	}

	virtual std::string describe() const {
	    std::string desc = "union(";
	    for (u32 i = 0; i < nLats_; ++i)
		desc += lats_[i]->describe() + ",";
	    desc.at(desc.size() - 1) = ')';
	    return desc;
	}
    };

    ConstLatticeRef unite(const ConstLatticeRefList &lats, ConstSemiringRef semiring) {
	switch (lats.size()) {
	case 0:
	    return ConstLatticeRef();
	case 1:
	    if (semiring)
		return changeSemiring(lats[0], semiring);
	    else
		return lats[0];
	default:
	    if (!semiring) {
		semiring = lats[0]->semiring();
		for (u32 i = 1; i < lats.size(); ++i)
		    if (!Semiring::equal(semiring, lats[0]->semiring()))
			Core::Application::us()->criticalError(
			    "Lattice union requires equal semirings, but \"%s\" != \"%s\".",
			    semiring->name().c_str(), lats[i]->semiring()->name().c_str());
	    }
	    return ConstLatticeRef(new UnionLattice(lats, semiring));
	}
    }

    ConstLatticeRef unite(ConstLatticeRef l1, ConstLatticeRef l2, ConstSemiringRef semiring) {
	ConstLatticeRefList lats(2);
	lats[0] = l1;
	lats[1] = l2;
	return unite(lats, semiring);
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    class UnionNode : public Node {
    private:
	u32 n_;
	ConstSemiringRef unionSemiring_;
	ConstLatticeRef union_;

    private:
	void buildUnion() {
	    if (!union_) {
		ConstLatticeRefList lats(n_);
		for (u32 i = 0; i < n_; ++i)
		    lats[i] = requestLattice(i);
		union_ = unite(lats, unionSemiring_);
	    }
	}

    public:
	UnionNode(const std::string &name, const Core::Configuration &config) :
	    Node(name, config),
	    n_(0) {}
	virtual ~UnionNode() {}

	virtual void init(const std::vector<std::string> &arguments) {
	    for (n_ = 0; connected(n_); ++n_);
	    if (n_ == 0)
		criticalError("At least one incoming lattice at port 0 required.");
	    Core::Component::Message msg = log();
	    if (n_ > 1)
		msg << "Combine " << n_ << " lattices.\n\n";
	    unionSemiring_ = Semiring::create(select("semiring"));
	    if (unionSemiring_)
		msg << "Union semiring:\n\t" << unionSemiring_->name();
	}

	virtual void finalize() {}

	virtual ConstLatticeRef sendLattice(Port to) {
	    verify(to == 0);
	    buildUnion();
	    return union_;
	}

	virtual void sync() {
	    union_.reset();
	}
    };
    NodeRef createUnionNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new UnionNode(name, config));
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    namespace {
	struct MeshedFullBoundaryBuilder {
	    typedef Fsa::Hash<Boundary, Boundary::Hash, Boundary::Equal> HashList;
	    Boundary operator()(const Boundary &b) const {
		return b;
	    }
	};

	struct MeshedTimeBoundaryBuilder {
	    struct BoundaryTimeHash {
		size_t operator() (const Boundary &b) const {
		    return size_t(b.time());
		}
	    };
	    struct BoundaryTimeEqual {
		bool operator() (const Boundary &b1, const Boundary &b2) const {
		    return b1.time() == b2.time();
		}
	    };
	    typedef Fsa::Hash<Boundary, Boundary::Hash, Boundary::Equal> HashList;
	    Boundary operator()(const Boundary &b) const {
		return Boundary(b.time());
	    }
	};
    } // namespace

    template<class MeshedBoundaryBuilder>
    ConstLatticeRef buildMesh(const ConstLatticeRefList &lats, ConstSemiringRef semiring, const MeshedBoundaryBuilder &meshedBoundaryBuilder = MeshedBoundaryBuilder()) {
	StaticBoundaries *b = new StaticBoundaries;
	StaticLattice *s = new StaticLattice(Fsa::TypeAcceptor);
	s->setProperties(lats[0]->knownProperties(), lats[0]->properties());
	s->setInputAlphabet(lats[0]->getInputAlphabet());
	s->setSemiring(semiring);
	s->setBoundaries(ConstBoundariesRef(b));
	s->addProperties(Fsa::PropertySortedByInputAndTarget);
	Core::Vector<Fsa::StateId> initialSids;
	typedef typename MeshedBoundaryBuilder::HashList BoundaryHashList;
	BoundaryHashList meshSids;
	for (u32 i = 0; i < lats.size(); ++i) {
	    ConstLatticeRef l = lats[i];
	    ConstStateMapRef topologicalSort = sortTopologically(l);
	    verify(topologicalSort);
	    const Boundary initialB = meshedBoundaryBuilder(l->boundary(topologicalSort->front()));
	    std::pair<Fsa::StateId, bool> meshInitial = meshSids.insertExisting(initialB);
	    if (!meshInitial.second) {
		State *meshInitialSp = new State(meshInitial.first);
		s->setState(meshInitialSp);
		initialSids.push_back(meshInitialSp->id());
		b->set(meshInitialSp->id(), initialB);
	    }
	    for (u32 j = 0; j < topologicalSort->size(); ++j) {
		Fsa::StateId sid = (*topologicalSort)[j];
		ConstStateRef sr = l->getState(sid);
		const Boundary meshB = meshedBoundaryBuilder(l->boundary(sid));
		Fsa::StateId meshSid = meshSids.find(meshB);
		verify(meshSid != BoundaryHashList::InvalidCursor);
		State *meshSp = s->fastState(meshSid);
		if (sr->isFinal()) {
		    if (meshSp->isFinal())
			meshSp->weight_ = semiring->collect(meshSp->weight(), sr->weight());
		    else
			meshSp->setFinal(sr->weight());
		}
		for (State::const_iterator a = sr->begin(); a != sr->end(); ++a) {
		    const Boundary targetB = meshedBoundaryBuilder(l->boundary(a->target()));
		    std::pair<Fsa::StateId, bool> meshTarget = meshSids.insertExisting(targetB);
		    if (!meshTarget.second) {
			State *meshTargetSp = new State(meshTarget.first);
			s->setState(meshTargetSp);
			b->set(meshTargetSp->id(), targetB);
		    } else
			if ((meshB.time() >= targetB.time()) && (meshSp->id() != meshTarget.first))
			    Core::Application::us()->warning(
				"Lattice \"%s\" contains null/negative-length arcs; meshed lattice might be cyclic.",
				l->describe().c_str());
		    /*
		      Arc meshA(meshTarget.first, a->weight(), a->input(), a->output());
		      State::iterator pos = meshSp->lower_bound(
		      meshA, Ftl::byInputAndTarget<Lattice>());
		      if ((pos == meshSp->end()) || (meshA.target() != pos->target())
		      || (meshA.input() != pos->input()))
		      meshSp->insert(pos, meshA);
		      else
		      pos->setWeight(semiring->collect(pos->weight(), meshA.weight()));
		    */
		    if (meshSp->id() != meshTarget.first)
			meshSp->newArc(meshTarget.first, a->weight(), a->input(), a->output());
		}
	    }
	}
	if (initialSids.size() == 1)
	    s->setInitialStateId(initialSids.front());
	else {
	    State *meshInitialSp;
	    Boundary initialB(0, Boundary::Transit(Bliss::Phoneme::term, Bliss::Phoneme::term));
	    std::pair<Fsa::StateId, bool> meshInitial = meshSids.insertExisting(initialB);
	    if (!meshInitial.second) {
		meshInitialSp = new State(meshInitial.first);
		s->setState(meshInitialSp);
		b->set(meshInitialSp->id(), initialB);
	    } else
		meshInitialSp = s->fastState(meshInitial.first);
	    for (Core::Vector<Fsa::StateId>::const_iterator itSid = initialSids.begin();
		 itSid != initialSids.end(); ++itSid)
		if (*itSid != meshInitialSp->id())
		    meshInitialSp->newArc(*itSid, semiring->one(), Fsa::Epsilon, Fsa::Epsilon);
	    s->setInitialStateId(meshInitialSp->id());
	}
	std::string desc = "mesh(" + lats[0]->describe();
	for (u32 i = 1; i < lats.size(); ++i)
	    desc += "," + lats[i]->describe();
	desc.at(desc.size() - 1) = ';';
	desc += semiring->name() + ")";
	s->setDescription(desc);
	return ConstLatticeRef(s);
    }

    ConstLatticeRef mesh(const ConstLatticeRefList &lats, ConstSemiringRef semiring, MeshType meshType) {
	if (lats.empty())
	    return ConstLatticeRef();
	if (!semiring) {
	    semiring = lats[0]->semiring();
	    for (u32 i = 1; i < lats.size(); ++i)
		if ((semiring.get() != lats[i]->semiring().get()) &&
		    !(semiring == lats[i]->semiring()))
		    Core::Application::us()->criticalError(
			"mesh: Mesh requires equal semirings, but \"%s\" != \"%s\"",
			semiring->name().c_str(), lats[i]->semiring()->name().c_str());
	}
	switch (meshType) {
	case MeshTypeFullBoundary:
	    return buildMesh<MeshedFullBoundaryBuilder>(lats, semiring);
	case MeshTypeTimeBoundary:
	    return buildMesh<MeshedTimeBoundaryBuilder>(lats, semiring);
	default:
	    defect();
	}
	return ConstLatticeRef();
    }

    ConstLatticeRef mesh(ConstLatticeRef l1, ConstLatticeRef l2, ConstSemiringRef semiring, MeshType meshType) {
	ConstLatticeRefList lats(2);
	lats[0] = l1;
	lats[1] = l2;
	return mesh(lats, semiring, meshType);
    }

    ConstLatticeRef mesh(ConstLatticeRef l, MeshType meshType) {
	ConstLatticeRefList lats(1, l);
	return mesh(lats, l->semiring(), meshType);
    }
    // -------------------------------------------------------------------------


    // -------------------------------------------------------------------------
    class MeshNode : public FilterNode {
    public:
	static const Core::Choice choiceMeshType;
	static const Core::ParameterChoice paramMeshType;
    private:
	MeshType meshType_;
	ConstLatticeRef meshL_;
    protected:
	ConstLatticeRef filter(ConstLatticeRef l) {
	    if (!l)
		return ConstLatticeRef();
	    if (!meshL_)
		meshL_ = mesh(l, meshType_);
	    return meshL_;
	}
    public:
	MeshNode(const std::string &name, const Core::Configuration &config) :
	    FilterNode(name, config) {}
	virtual ~MeshNode() {}

	virtual void init(const std::vector<std::string> &arguments) {
	    Core::Component::Message msg = log();
	    Core::Choice::Value meshType = paramMeshType(config);
	    if (meshType ==  Core::Choice::IllegalValue)
		criticalError("Unknown mesh type");
	    meshType_ = MeshType(meshType);
	    msg << "mesh type is \"" << choiceMeshType[meshType_] << "\"\n";
	}

	virtual void sync() {
	    meshL_.reset();
	}
    };
    const Core::Choice MeshNode::choiceMeshType(
	"full", MeshTypeFullBoundary,
	"time", MeshTypeTimeBoundary,
	Core::Choice::endMark());
    const Core::ParameterChoice MeshNode::paramMeshType(
	"mesh-type",
	&MeshNode::choiceMeshType,
	"type of mesh",
	MeshTypeFullBoundary);
    NodeRef createMeshNode(const std::string &name, const Core::Configuration &config) {
	return NodeRef(new MeshNode(name, config));
    }
    // -------------------------------------------------------------------------

} // namespace Flf
