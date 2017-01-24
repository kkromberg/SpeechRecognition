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
#include "EigenTransform.hh"
#include <Math/Module.hh>

using namespace Signal;

//----------------------------------------------------------------------------
const Core::ParameterInt EigenTransform::paramReducedDimension(
    "reduced-dimesion", "projected dimesion, if 0, dimension is not changed.", 0, 0);

const Core::ParameterFloat EigenTransform::paramReducedDimensionByThreshold(
    "reduced-dimesion-threshold", "projected dimesion depending on eigenwert values which will be reduced if lower than threshold; if 0, dimension is not changed.", 0.0, 0.0);

EigenTransform::EigenTransform(const Core::Configuration &configuration) :
    Core::Component(configuration),
    Precursor(configuration, "projector-matrix"),
    resultsChannel_(configuration, "results", Core::Channel::disabled)
{}

EigenTransform::~EigenTransform()
{}

bool EigenTransform::work()
{
    bool success = true;
    if (!createProjector()) success = false;
    if (!Precursor::work()) success = false;
    writeResults();
    return success;
}

bool EigenTransform::createProjector()
{
    bool success = true;
    size_t reducedDimension = paramReducedDimension(config);
    f32 reducedDimensionThreshold = paramReducedDimensionByThreshold(config);

    if ((reducedDimension != 0) && (reducedDimensionThreshold != 0.0)) {
	error("Projected-dimension must be defined by either 'reduced-dimesion' or 'reduced-dimesion-threshold', not both");
	reducedDimensionThreshold=0.0;
    }
    if(reducedDimensionThreshold != 0.0) {
	reducedDimension = eigenvectors_.nColumns();
	for(size_t i=0; i<eigenvectors_.nColumns();++i) {
	    if (eigenvalues_[i] < reducedDimensionThreshold) --reducedDimension;
	}
	//         std::cout << "*** debug: reduced by threshold " << reducedDimensionThreshold
	//                   << " from dimension " << eigenvectors_.nColumns()
	//                   << " to " << reducedDimension
	//                   << std::endl;
	if(reducedDimension==0)
	    warning("Projected-dimension was reduced to zero, i.e. reduction will be disabled. "\
		    "Check your threshold setting as this is not that what you wanted.");
	ensure(reducedDimension >= 0);
    }

    if (reducedDimension == 0) reducedDimension = eigenvectors_.nColumns();
    if (reducedDimension > eigenvectors_.nColumns()) {
	error("Projected-dimension (%zd) is larger than number of eigenvectors (%zd).",
	      reducedDimension, eigenvectors_.nColumns());
	reducedDimension = eigenvectors_.nColumns();
	success = false;
    }
    transform_ = eigenvectors_.transpose();
    transform_.resize(reducedDimension, transform_.nColumns());
    return success;
}

void EigenTransform::writeResults(
    Core::XmlWriter &os) const
{
    os << Core::XmlOpen("eigen-transform");
    os << Core::XmlOpen("eigenvalues");
    os << eigenvalues_;
    os << Core::XmlClose("eigenvalues");
    os << Core::XmlOpen("eigenvectors");
    os << eigenvectors_;
    os << Core::XmlClose("eigenvectors");
    os << Core::XmlClose("eigen-transform");
}

//----------------------------------------------------------------------------
PrincipalComponentAnalysis::PrincipalComponentAnalysis(const Core::Configuration &configuration) :
    Core::Component(configuration),
    Precursor(configuration),
    eigenvalueProblem_(0),
    covarianceMatrixChannel_(configuration, "covariance-matrix", Core::Channel::disabled)
{
    eigenvalueProblem_ = Math::Module::instance().createEigenvalueProblem(select("eigenvalue-problem"));
    if (!eigenvalueProblem_)
	criticalError("Failed to initialize the eigenvalue problem.");
}

PrincipalComponentAnalysis::~PrincipalComponentAnalysis()
{
    delete eigenvalueProblem_;
}


bool PrincipalComponentAnalysis::work(const ScatterMatrix &covarianceMatrix)
{
    verify(eigenvalueProblem_);

    eigenvalues_.clear();
    eigenvectors_.clear();

    if (covarianceMatrixChannel_.isOpen())
	writeScatterMatrix(covarianceMatrixChannel_, totalScatterType, covarianceMatrix);
    bool success = true;
    if (!eigenvalueProblem_->solveSymmetricAndFinalize(
	    covarianceMatrix, eigenvalues_, eigenvectors_)) success = false;
    if (!Precursor::work()) success = false;
    return success;
}

bool PrincipalComponentAnalysis::work()
{
    ScatterMatrix covarianceMatrix;
    return (readScatterMatrix(paramTotalScatterFilename(config),
			      totalScatterType, covarianceMatrix) &&
	    work(covarianceMatrix));
}

//----------------------------------------------------------------------------
LinearDiscriminantAnalysis::LinearDiscriminantAnalysis(const Core::Configuration &configuration) :
    Core::Component(configuration),
    Precursor(configuration),
    generalizedEigenvalueProblem_(0),
    eigenvalueProblem_(0),
    betweenClassScatterMatrixChannel_(configuration, "between-class-scatter-matrix", Core::Channel::disabled),
    withinClassScatterMatrixChannel_(configuration, "within-class-scatter-matrix", Core::Channel::disabled)
{
    generalizedEigenvalueProblem_ =
	Math::Module::instance().createGeneralizedEigenvalueProblem(
	    select("generalized-eigenvalue-problem"));
    if (!generalizedEigenvalueProblem_)
	criticalError("Failed to initialize the generalized eigenvalue problem.");

    if (betweenClassScatterMatrixChannel_.isOpen() ||
	withinClassScatterMatrixChannel_.isOpen()) {
	eigenvalueProblem_ =
	    Math::Module::instance().createEigenvalueProblem(
		select("eigenvalue-problem"));
	if (!eigenvalueProblem_)
	    criticalError("Failed to initialize the eigenvalue problem.");
    }
}

LinearDiscriminantAnalysis::~LinearDiscriminantAnalysis()
{
    delete generalizedEigenvalueProblem_;
    delete eigenvalueProblem_;
}

bool LinearDiscriminantAnalysis::work(const ScatterMatrix &betweenClassScatterMatrix,
				      const ScatterMatrix &withinClassScatterMatrix)
{
    verify(generalizedEigenvalueProblem_);

    eigenvalues_.clear();
    eigenvectors_.clear();

    if (betweenClassScatterMatrixChannel_.isOpen()) {
	writeScatterMatrix(betweenClassScatterMatrixChannel_,
			   betweenClassScatterType,
			   betweenClassScatterMatrix,
			   eigenvalueProblem_);
    }
    if (withinClassScatterMatrixChannel_.isOpen()) {
	writeScatterMatrix(withinClassScatterMatrixChannel_,
			   withinClassScatterType,
			   withinClassScatterMatrix,
			   eigenvalueProblem_);
    }

    bool success = true;
    if (!generalizedEigenvalueProblem_->solveSymmetricAndFinalize(
	    betweenClassScatterMatrix, withinClassScatterMatrix,
	    eigenvalues_, eigenvectors_)) success = false;
    if (!Precursor::work()) success = false;
    return success;
}

bool LinearDiscriminantAnalysis::work()
{
    ScatterMatrix betweenClassScatterMatrix;
    ScatterMatrix withinClassScatterMatrix;
    return (readScatterMatrix(paramBetweenClassScatterFilename(config),
			      betweenClassScatterType, betweenClassScatterMatrix) &&
	    readScatterMatrix(paramWithinClassScatterFilename(config),
			      withinClassScatterType, withinClassScatterMatrix) &&
	    work(betweenClassScatterMatrix, withinClassScatterMatrix));
}
