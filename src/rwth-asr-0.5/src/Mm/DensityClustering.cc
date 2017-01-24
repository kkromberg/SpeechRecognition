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
#include <Mm/DensityClustering.hh>

using namespace Mm;

const Core::ParameterInt DensityClusteringBase::paramNumClusters(
    "clusters", "number of density clusters to build for density preselection", 256, 1, 256);
const Core::ParameterInt DensityClusteringBase::paramSelectClusters(
    "select-clusters",
    "number of clusters to select in density preselection."
    "when it equals the total number of clusters, no preselection is performed.", 32, 1, 256);
const Core::ParameterString DensityClusteringBase::paramFile(
    "file", "file where to cache the clustering of the density preselection", "");
const Core::ParameterInt DensityClusteringBase::paramClusteringIterations(
    "iterations", "number of clustering iterations", 5);
const Core::ParameterFloat DensityClusteringBase::paramBackoffScore(
    "backoff-score", "score used if no cluster is selected", 40000);

const std::string DensityClusteringBase::FileMagic = "SPRINT-DC";
const u32 DensityClusteringBase::FileFormatVersion = 2;

DensityClusteringBase::DensityClusteringBase(const Core::Configuration &config) :
	Core::Component(config),
	nClusters_(paramNumClusters(config)),
	nSelected_(paramSelectClusters(config)),
	dimension_(0), nDensities_(0),
	backoffScore_(paramBackoffScore(config)) {}


void DensityClusteringBase::init(u32 dimension, u32 nDensities)
{
    dimension_ = dimension;
    nDensities_ = nDensities;
    clusterIndexForDensity_.resize(nDensities_, 0);
    // verification to make sure that ClusterIndex can hold the cluster-indices
    verify(nClusters_ - 1 <= Core::Type<ClusterIndex>::max);
    verify(nClusters_ <= nDensities_);
    verify(nSelected_ <= nClusters_);
}


bool DensityClusteringBase::load(const std::string &filename)
{
    Core::BinaryInputStream is(filename);
    if (!is.isOpen())
	return false;
    std::string magic;
    u32 version;
    is >> magic >> version;
    if (magic != FileMagic || version != FileFormatVersion) {
	warning("wrong file format. expected '%s %d' found '%s %d'",
		FileMagic.c_str(), FileFormatVersion, magic.c_str(), version);
	return false;
    }
    if (!readTypes(is)) {
	warning("cannot read type information");
	return false;
    }
    u32 dimension, clusters, densities;
    is >> dimension >> clusters >> densities;
    if (dimension != dimension_ || clusters != nClusters_ || densities != nDensities_) {
	warning("cached density clustering does not match current settings");
	return false;
    }
    is >> clusterIndexForDensity_;
    return readMeans(is);
}

bool DensityClusteringBase::write(const std::string &filename) const
{
    Core::BinaryOutputStream os(filename);
    if (!os.isOpen())
	return false;
    os << FileMagic << FileFormatVersion;
    if (!writeTypes(os))
	return false;
    os << dimension_ << nClusters_ << nDensities_;
    os << clusterIndexForDensity_;
    if (!writeMeans(os))
	return false;
    return os.good();
}
