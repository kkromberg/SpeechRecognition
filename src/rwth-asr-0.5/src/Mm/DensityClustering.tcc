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
#include <set>
#include <algorithm>
#include <functional>
#include <Core/BinaryStream.hh>
#include <Core/Utility.hh>
#include <Core/Extensions.hh>
#include <Mm/Utilities.hh>

namespace Mm {

template<class F, class D>
bool DensityClustering<F, D>::writeMeans(Core::BinaryOutputStream &os) const
{
    std::copy(clusterMeans_, clusterMeans_ + nClusters_ * dimension_,
              Core::BinaryOutputStream::Iterator<FeatureType>(os));
    return os.good();
}

template<class F, class D>
bool DensityClustering<F, D>::readMeans(Core::BinaryInputStream &is)
{
    FeatureType *m = clusterMeans_;
    for (u32 i = 0; i < nClusters_ * dimension_; ++i, ++m)
	is >> *m;
    return !is.fail();
}

template<class F, class D>
bool DensityClustering<F, D>::writeTypes(Core::BinaryOutputStream &os) const
{
    os << Core::Type<FeatureType>::name << Core::Type<DistanceType>::name;
    return os.good();
}

template<class F, class D>
bool DensityClustering<F, D>::readTypes(Core::BinaryInputStream &is) const
{
    std::string featureType, distanceType;
    is >> featureType >> distanceType;
    if (is.fail()) return false;
    return (featureType == Core::Type<FeatureType>::name &&
	    distanceType == Core::Type<DistanceType>::name);
}


/**
 * Initialize each cluster with a unique random density.
 * Updates clusterMeans_
 */
template<class F, class D>
void DensityClustering<F, D>::initializeClusters(const FeatureType *densities)
{
    std::set<u32> usedDensities;
    srand(1); // Pseudo-random
    for (u32 cluster = 0; cluster < nClusters_; ++cluster) {
	u32 initializeWithDensity = 0;
	do {
	    initializeWithDensity = rand() % nDensities_;
	} while (usedDensities.count(initializeWithDensity));
	usedDensities.insert(initializeWithDensity);
	const FeatureType *mean = meanForDensity(densities, initializeWithDensity);
	std::copy(mean, mean + dimension_, meanForCluster(cluster));
    }
}

/**
 * Assign densities to clusters.
 * Updates clusterIndexForDensity_
 */
template<class F, class D>
void DensityClustering<F, D>::assignDensities(
	const FeatureType *densities, DensityAssignment &densityAssignment)
{
    for (u32 density = 0; density < nDensities_; ++density) {
	DistanceType bestDistance = Core::Type<DistanceType>::max;
	u32 bestCluster = 0;
	for (u32 cluster = 0; cluster < nClusters_; ++cluster) {
	    DistanceType dist = unrolledVectorDistance<FeatureType, DistanceType>(
		    meanForCluster(cluster), meanForDensity(densities, density), dimension_);
	    if (dist < bestDistance) {
		bestDistance = dist;
		bestCluster = cluster;
	    }
	}
	verify(bestDistance != Core::Type<DistanceType>::max);
	clusterIndexForDensity_[density] = bestCluster;
	densityAssignment[bestCluster].push_back(density);
    }
}

template<class F, class D>
f64 DensityClustering<F, D>::updateClusterMeans(
	const FeatureType *densities, DensityAssignment &densityAssignment)
{
    f64 totalAssignmentDistance = 0;
    for (u32 cluster = 0; cluster < nClusters_; ++cluster) {
	const std::vector<u32>& assignedDensities(densityAssignment[cluster]);
	if (assignedDensities.empty())
	    continue;
	std::vector<f64> sums(dimension_, 0);
	for (u32 assigned = 0; assigned < assignedDensities.size(); ++assigned) {
	    const FeatureType *density = meanForDensity(densities, assignedDensities[assigned]);
	    std::transform(sums.begin(), sums.end(), density, sums.begin(), std::plus<f64>());
	}
	std::transform(sums.begin(), sums.end(), meanForCluster(cluster),
		       std::bind2nd(std::divides<f64>(), assignedDensities.size()));
	for (u32 assigned = 0; assigned < assignedDensities.size(); ++assigned)
	    totalAssignmentDistance += unrolledVectorDistance<FeatureType, DistanceType>(
		    meanForCluster(cluster),
		    meanForDensity(densities, assignedDensities[assigned]), dimension_);
    }
    return totalAssignmentDistance;
}

template<class F, class D>
void DensityClustering<F, D>::build(const FeatureType *densities)
{
    verify(nClusters_ > 0);
    verify(nDensities_ > 0);
    verify(dimension_ > 0);
    clusterMeans_ = new FeatureType[nClusters_ * dimension_];
    std::string cacheFile = paramFile(config);
    if (!cacheFile.empty()) {
	if (load(cacheFile)) {
	    log("using cached density clustering: ") << cacheFile;
	    return;
	} else {
	    warning("cannot read density clustering ") << cacheFile;
	}
    }
    Core::XmlChannel statChannel(config, "statistics");
    if (statChannel.isOpen()) {
	statChannel << Core::XmlOpen("density-clustering") +
		       Core::XmlAttribute("clusters", nClusters_);
    }
    verify(nClusters_ <= nDensities_);
    initializeClusters(densities);

    // Iteratively cluster the densities, and update the clusters
    const u32 iterations = paramClusteringIterations(config);
    for (u32 i = 0; i < iterations; ++i) {
	DensityAssignment densitiesAssignedToClusters(nClusters_);
	assignDensities(densities, densitiesAssignedToClusters);
	f64 distance = updateClusterMeans(densities, densitiesAssignedToClusters);
	if (statChannel.isOpen()) {
	    statChannel << Core::XmlOpen("total-distance") + Core::XmlAttribute("iteration", i)
		        << distance << Core::XmlClose("total-distance");
	}
    }
    if (statChannel.isOpen()) {
	statChannel << Core::XmlClose("density-clustering");
    }
    if (!cacheFile.empty() && write(cacheFile)) {
	log("density clustering written: ") << cacheFile;
    }
}

template<class F, class D>
void DensityClustering<F, D>::selectClusters(bool *selection, FeatureType* feature) const
{
    typedef std::pair<DistanceType, u32> SortedClusterItem;
    SortedClusterItem clustersByDistance[nClusters_];
    // Compute the distance for each cluster
    for (u32 cluster = 0; cluster < nClusters_; ++cluster) {
	DistanceType d = unrolledVectorDistance<FeatureType, DistanceType>(
		feature, meanForCluster(cluster), dimension_);
	clustersByDistance[cluster] = std::make_pair(d, cluster);
    }

    // Sort the clustersByDistance array by distance
    // std::partial_sort would be an alternative here,
    // but for the small array introsort is faster.
    std::sort(clustersByDistance, clustersByDistance + nClusters_,
              Core::composeBinaryFunction(std::less<DistanceType>(),
                                          Core::select1st<SortedClusterItem>(),
                                          Core::select1st<SortedClusterItem>()));

    // Mark the best clusters as selected, the others as unselected
    std::fill(selection, selection + nClusters_, false);
    for(u32 sortedCluster = 0; sortedCluster < nSelected_; ++sortedCluster) {
	*(selection + clustersByDistance[sortedCluster].second) = true;
    }
}


}  // namespace Mm
