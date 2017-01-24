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
#ifndef _SIGNAL_EXTERNAL_VECTOR_NORMALIZATION_HH
#define _SIGNAL_EXTERNAL_VECTOR_NORMALIZATION_HH


#include <Core/Parameter.hh>
#include <Core/Utility.hh>

#include <fstream>
#include <vector>
#include <Math/Vector.hh>
#include <Core/Types.hh>
#include <Core/VectorParser.hh>
#include <Core/Hash.hh>
#include <Flow/Data.hh>
#include <Flow/Node.hh>
#include <Flow/Timestamp.hh>
#include <Flow/Types.hh>
#include <Flow/Vector.hh>
#include <Math/Module.hh>
#include "Node.hh"

#include <Core/VectorParser.hh>

namespace Signal {
const Core::ParameterString paramExternalVectorFileName(
	"file", "name of external vector file to load");

/** ExternalVectorAddition */
template <class T>
class ExternalVectorAddition {
public:
    typedef T Value;

    static std::string name() {
	return Core::Type<Value>::name + std::string("-add");
    }

    void operator()(std::vector<Value>& v, Flow::Vector<Value>& u) {
	// do the operation
	typename Flow::Vector<Value>::iterator itu = u.begin();
	for (typename std::vector<Value>::iterator itv = v.begin();
		itv != v.end(); itv++, itu++) {
	    *itv += *itu;
	}
    }
};

/** ExternalVectorSubstraction */
template <class T>
class ExternalVectorSubtraction {
public:
    typedef T Value;

    static std::string name() {
	return Core::Type<Value>::name + std::string("-sub");
    }

    void operator()(std::vector<Value>& v, Flow::Vector<Value>& u) {
	// do the operation
	typename Flow::Vector<Value>::iterator itu = u.begin();
	for (typename std::vector<Value>::iterator itv = v.begin();
		itv != v.end(); itv++, itu++) {
	    *itv -= *itu;
	}
    }
};

/** ExternalVectorMultiplication */
template <class T>
class ExternalVectorMultiplication {
public:
    typedef T Value;

    static std::string name() {
	return Core::Type<Value>::name + std::string("-mul");
    }

    void operator()(std::vector<Value>& v, Flow::Vector<Value>& u) {
	// do the operation
	typename Flow::Vector<Value>::iterator itu = u.begin();
	for (typename std::vector<Value>::iterator itv = v.begin();
		itv != v.end(); itv++, itu++) {
	    *itv *= *itu;
	}
    }
};

/** ExternalVectorDivision */
template <class T>
class ExternalVectorDivision {
public:
    typedef T Value;

    static std::string name() {
	return Core::Type<Value>::name + std::string("-div");
    }

    void operator()(std::vector<Value>& v, Flow::Vector<Value>& u) {
	// do the operation
	typename Flow::Vector<Value>::iterator itu = u.begin();
	for (typename std::vector<Value>::iterator itv = v.begin();
		itv != v.end(); itv++, itu++) {
	    *itv /= *itu;
	}
    }
};

/** ExternalVectorOperationNode
  *
  * load a vector from file and do an operation
  */
template<class ArithmeticFunction>
class ExternalVectorFunctionNode : public Flow::SleeveNode {
private:
    ArithmeticFunction operation_;

    Flow::Vector<typename ArithmeticFunction::Value> vector_;
    const Core::Configuration config_;
public:
    static std::string filterName() {
	return "signal-vector-" + ArithmeticFunction::name() + "-external";
    }

    ExternalVectorFunctionNode(const Core::Configuration &c);
    virtual ~ExternalVectorFunctionNode();

    virtual bool configure();
    virtual bool setParameter(const std::string &name, const std::string &value);
    virtual bool work(Flow::PortId p);

private:
    Flow::Vector<typename ArithmeticFunction::Value>* loadVector(const std::string &filename) {

	// check for empty filename
	if (filename.empty()) {
	    error() << "Vector filename is empty.";
	} else {
	    Core::XmlVectorDocument<float> parser(config, vector_);
	    parser.parseFile(filename.c_str());

	    log() << vector_;
	}
	return &vector_;
    }
};


template<class ArithmeticFunction>
ExternalVectorFunctionNode<ArithmeticFunction>::ExternalVectorFunctionNode(const Core::Configuration &c) : Core::Component(c), SleeveNode(c) {
    config = c;
    loadVector(paramExternalVectorFileName(config));
};

template<class ArithmeticFunction>
ExternalVectorFunctionNode<ArithmeticFunction>::~ExternalVectorFunctionNode() {
}

template<class ArithmeticFunction>
bool ExternalVectorFunctionNode<ArithmeticFunction>::work(Flow::PortId p) {
    Flow::DataPtr<Flow::Vector<typename ArithmeticFunction::Value> > in;
    if (! getData(0, in))
	return SleeveNode::putData(0, in.get());

    in.makePrivate();

    /* check the size of the two vectors */
    s32 sizeIn =(s32) (*in).size();
    s32 sizeVec=(s32) vector_.size();
    if (sizeIn != sizeVec) {
	//			warning() << "Input Vector dimension mismatch: is " << sizeIn << " should be " << sizeVec << ".\n";
	if (sizeIn > sizeVec) {
	    error() << "Input Vector is to long, should be " << sizeVec << " is " << sizeIn << "\n.";
	}
    }

    operation_(*in, vector_);
    return putData(0, in.get());
}

/* set the configuration parameter */
template<class ArithmeticFunction>
bool ExternalVectorFunctionNode<ArithmeticFunction>::setParameter(const std::string &name, const std::string &value) {
    bool retrVal = false;

    /* test if parameter matches */
    if (paramExternalVectorFileName.match(name)) {
	loadVector(paramMatrixMultiplicationFileName(value));
	retrVal = true;
    }

    return retrVal;
}

/* configuration function */
template<class ArithmeticFunction>
bool ExternalVectorFunctionNode<ArithmeticFunction>::configure() {
    Core::Ref<const Flow::Attributes> a = getInputAttributes(0);

    if (!configureDatatype(a, Flow::Vector<typename ArithmeticFunction::Value>::type()))
	return false;

    return putOutputAttributes(0, a);
}

} // _NAMESPACE_signal

#endif // _SIGNAL_EXTERNAL VECTOR_NORMALIZATION_HH
