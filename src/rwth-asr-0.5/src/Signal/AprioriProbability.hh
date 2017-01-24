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
#ifndef _SIGNAL_APRIORI_PROBABILITY_HH
#define _SIGNAL_APRIORI_PROBABILITY_HH

#include <Flow/Node.hh>

namespace Signal {

    /** Base class for a-priory probabity. */
    class AprioriProbability  : public virtual Core::Component {
    public:
	typedef f32 Score;
    public:
	AprioriProbability(const Core::Configuration &c) : Component(c) {}
	virtual ~AprioriProbability() {}

	virtual bool setClasses(const std::vector<std::string> &classLabels) = 0;

	/** @return is the score -log p(classIndex). */
	virtual Score operator[](u32 classIndex) = 0;
    };

    /** Uniform class a-priory probabity. */
    class UniformAprioriProbability  : public AprioriProbability {
    private:
	size_t nClasses_;
	Score logNClasses_;
    public:
	UniformAprioriProbability(const Core::Configuration &c) :
	    Component(c), AprioriProbability(c), nClasses_(0), logNClasses_(0) {}
	virtual ~UniformAprioriProbability() {}

	virtual bool setClasses(const std::vector<std::string> &classLabels) {
	    logNClasses_ = std::log((Score)(nClasses_ = classLabels.size())); return true;
	}

	virtual Score operator[](u32 classIndex) { require_(classIndex < nClasses_); return logNClasses_; }
    };

} // namespace Signal

#endif // _SIGNAL_APRIORI_PROBABILITY_HH
