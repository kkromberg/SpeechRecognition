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
#ifndef _MM_FEATURESCORER_HH
#define _MM_FEATURESCORER_HH

#include "Types.hh"
#include "Feature.hh"
#include "Utilities.hh"
#include <vector>
#include <Core/Component.hh>
#include <Core/Dependency.hh>

namespace Mm {

    /** Abstract feature scorer interface. */
    class FeatureScorer :
	public virtual Core::Component,
	public Core::ReferenceCounted
    {
    protected:
	/** Implement emission independent precalculations for feature vector */
	class ContextScorer : public Core::ReferenceCounted {
	protected:
	    ContextScorer() {}
	public:
	    virtual ~ContextScorer() {}

	    virtual EmissionIndex nEmissions() const = 0;
	    virtual Score score(EmissionIndex e) const = 0;
	};
	friend class ContextScorer;
    public:
	FeatureScorer(const Core::Configuration &c) : Core::Component(c) {}
	virtual ~FeatureScorer() {}

	virtual EmissionIndex nMixtures() const = 0;
	virtual void getFeatureDescription(FeatureDescription &description) const = 0;
	virtual void getDependencies(Core::DependencySet &dependencies) const {
	    FeatureDescription description(*this);
	    getFeatureDescription(description);
	    description.getDependencies(dependencies);
	}

	typedef Core::Ref<const ContextScorer> Scorer;
	virtual Scorer getScorer(Core::Ref<const Feature>) const = 0;
	virtual Scorer getScorer(const FeatureVector &) const = 0;

	/**
	 * reset should be overloaded/defined in/for
	 * featurescorer related to sign language recognition
	 * especially the tracking part
	 *
	 */
	virtual void reset() const {};

	/**
	 * setSegmentName should be overloaded/defined in classes
	 * using embedded flow networks to create unambigious ids
	 * for cache nodes
	 */
	virtual void setSegmentName(const std::string name) const {};

	/**
	 * finalize should be overloaded/defined in classes using
	 * embedded flow networks to sent final end of sequence token
	 * if neccessary
	 */
	virtual void finalize() const {};

	/**
	 * Return true if the feature scorer buffers features.
	 */
	virtual bool isBuffered() const { return false; }

	/**
	 * Add a feature to the feature buffer.
	 * Implementation required if isBuffered() == true
	 */
	virtual void addFeature(const FeatureVector &f) const {}
	virtual void addFeature(Core::Ref<const Feature> f) const {}

	/**
	 * Return a scorer for the current feature without adding a
	 * new feature to the buffer.
	 * Should be called until bufferEmpty() == true.
	 * Requires bufferEmpty() == false.
	 * Implementation required if isBuffered() == true
	 */
	virtual Scorer flush() const { return Scorer(); }

	/**
	 * Return true if the feature buffer is full.
	 * Implementation required if isBuffered() == true
	 */
	virtual bool bufferFilled() const { return true; }

	/**
	 * Return true if the feature buffer is empty.
	 * Implementation required if isBuffered() == true
	 */
	virtual bool bufferEmpty() const { return true; }

	/**
	 * Return the number of buffered features required to
	 * execture getScorer().
	 * Implementation required if isBuffered() == true
	 */
	virtual u32 bufferSize() const { return 0; }
    };

    /** Abstract feature scorer interface with cached scores. */
    class CachedFeatureScorer : public FeatureScorer {
	typedef FeatureScorer Precursor;
    protected:
	/** Implement emission independent precalculations for feature vector */
	class CachedContextScorer : public ContextScorer {
	private:
	    const CachedFeatureScorer *featureScorer_;
	    mutable Cache<Score> cache_;
	protected:
	    CachedContextScorer(const CachedFeatureScorer *featureScorer, EmissionIndex nEmissions) :
		featureScorer_(featureScorer), cache_(nEmissions) {}
	public:
	    virtual ~CachedContextScorer() {}
	    EmissionIndex nEmissions() const { return cache_.size(); }
	    virtual Score score(EmissionIndex e) const {
		require_(0 <= e && e < nEmissions());
		if (!cache_.isCalculated(e))
		    return cache_.set(e, featureScorer_->calculateScore(this, e));
		return cache_[e];
	    }
	};
	virtual Score calculateScore(const CachedContextScorer *, MixtureIndex) const = 0;
    public:
	CachedFeatureScorer(const Core::Configuration &c) : Core::Component(c), Precursor(c) {}
	virtual ~CachedFeatureScorer() {}
    };

} // namespace Mm

#endif // _MM_FEATURESCORER_HH
