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
#include "CovarianceEstimator.hh"
#include <Math/Vector.hh>
#include <Math/Module.hh>

using namespace Speech;

CovarianceEstimator::CovarianceEstimator(const Core::Configuration &c)
    : Component(c), Extractor(c), Estimator(c), needResize_(true)
{}

void CovarianceEstimator::processFeature(Core::Ref<const Speech::Feature> feature)
{
    if (feature->nStreams() and feature->mainStream()->size()) {
	accumulate(*feature->mainStream());
    }
}

void CovarianceEstimator::setFeatureDescription(const Mm::FeatureDescription &description)
{
    description.verifyNumberOfStreams(1);
    size_t d;
    description.mainStream().getValue(Mm::FeatureDescription::nameDimension, d);
    if(needResize_) {
	setDimension(d);
	needResize_ = false;
    }
}

// Mean and diagonal covariance estimator

const Core::ParameterString MeanAndDiagonalCovarianceEstimator::paramVarianceFilename(
    "variance-file", "Output filename for variance vector");
const Core::ParameterString MeanAndDiagonalCovarianceEstimator::paramStandardDeviationFilename(
    "standard-deviation-file", "Output filename for standard deviation vector");
const Core::ParameterString MeanAndDiagonalCovarianceEstimator::paramMeanFilename(
    "mean-file", "Output filename for mean");
const Core::ParameterFloat MeanAndDiagonalCovarianceEstimator::paramElementThresholdMin(
    "element-threshold-min", "Min threshold for every variance entry", 0.0);
const Core::ParameterInt MeanAndDiagonalCovarianceEstimator::paramOutputPrecision(
    "output-precision", "Number of decimal digits in text output formats", 20);


MeanAndDiagonalCovarianceEstimator::MeanAndDiagonalCovarianceEstimator(const Core::Configuration &c)
    : Component(c), Extractor(c), squareSum_(0), sum_(0), count_(0), finalized_(false)
{}

void MeanAndDiagonalCovarianceEstimator::processFeature(Core::Ref<const Speech::Feature> feature)
{
    Math::Vector<f32> x = *feature->mainStream();
    Math::Vector<f32> xSquared;
    xSquared.squareVector(x);
    sum_ += x;
    squareSum_ += xSquared;
    ++count_;
}

void MeanAndDiagonalCovarianceEstimator::setFeatureDescription(const Mm::FeatureDescription &description)
{
    description.verifyNumberOfStreams(1);
    size_t d;
    description.mainStream().getValue(Mm::FeatureDescription::nameDimension, d);
    squareSum_.resize(d);
    sum_.resize(d);
}

void MeanAndDiagonalCovarianceEstimator::finalize(){
    require(count_ != 0);
    u32 n = sum_.size();
    mean_.resize(n);
    for (u32 i = 0; i < n; i++){
	mean_.at(i) = sum_.at(i) / count_;
    }
    sum_.resize(0);
    variance_.resize(n);
    f64 threshold = paramElementThresholdMin(config);
    for (u32 i = 0; i < n; i++){
	variance_.at(i) = squareSum_.at(i) / count_ - mean_.at(i) * mean_.at(i);
	if (variance_.at(i) < threshold){
	    variance_.at(i) = threshold;
	}
    }
    squareSum_.resize(0);
    finalized_ = true;
}

bool MeanAndDiagonalCovarianceEstimator::write(){
    if (!finalized_){
	finalize();
    }
    bool success = true;
    std::string meanFilename = paramMeanFilename(config);
    std::string varianceFilename = paramVarianceFilename(config);
    std::string standardDeviationFilename = paramStandardDeviationFilename(config);
    if (meanFilename != "" ){
	if (Math::Module::instance().formats().write(
		meanFilename, mean_, paramOutputPrecision(config)))
	    log("Mean vector written to '%s'.", meanFilename.c_str());
	else {
	    error("Failed to write mean to '%s'.", meanFilename.c_str());
	    success = false;
	}
    }
    if (varianceFilename != "" ){
	if (Math::Module::instance().formats().write(
		varianceFilename, variance_, paramOutputPrecision(config)))
	    log("Diagonal variance vector written to '%s'.", varianceFilename.c_str());
	else {
	    error("Failed to write diagonal variance to '%s'.", varianceFilename.c_str());
	    success = false;
	}
    }
    if (standardDeviationFilename != "" ){
	Math::Vector<f32> sigma(variance_);
	sigma.takeSquareRoot();
	if (Math::Module::instance().formats().write(
		standardDeviationFilename, sigma, paramOutputPrecision(config)))
	    log("Standard deviation vector written to '%s'.", standardDeviationFilename.c_str());
	else {
	    error("Failed to write standard deviation to '%s'.", standardDeviationFilename.c_str());
	    success = false;
	}
    }
    return success;
}
