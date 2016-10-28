/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Mixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "emmintrin.h"
#include "pmmintrin.h"

namespace {
  // compute -log(exp(-a) + exp(-b)) using the following equality:
  // -log(exp(-a) + exp(-b)) = -log(exp(-a-c) + exp(-b-c)) - c
  //                         = -log(1 + exp(-abs(a-b))) - c
  // where c = max(-a, -b) = -min(a, b)
  double logsum(double a, double b) {
    const double diff = a - b;
    // if a = b = inf then a - b = nan, but for logsum the result is well-defined (inf)
    if (diff != diff) {
      return std::numeric_limits<double>::infinity();
    }
    return -log1p(std::exp(-std::abs(diff))) + std::min(a, b);
  }

  size_t build_mapping(std::vector<size_t> const& refs, std::vector<size_t>& mapping) {
    mapping.resize(refs.size());

    size_t count = 0ul;
    for (size_t i = 0ul; i < refs.size(); i++) {
      if (refs[i] > 0ul) {
        mapping[i] = count;
        count++;
      }
    }

    return count;
  }

  void test(bool condition, char const* error_msg) {
    if (not condition) {
      std::cerr << error_msg << std::endl;
      abort();
    }
  }

  void read_accumulator(std::istream& in, size_t expected_dimension,
                        std::vector<size_t>& refs,
                        std::vector<double>& features,
                        std::vector<double>& weight) {
    uint32_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    test(in.gcount() == sizeof(size), "Error reading size");

    refs.resize(size);
    features.resize(size * expected_dimension);
    weight.resize(size);

    for (size_t i = 0ul; i < size; i++) {
      uint32_t dimension;

      in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
      test(in.gcount() == sizeof(dimension), "Error reading dimension");
      test(dimension == expected_dimension, "Invalid dimension");

      in.read(reinterpret_cast<char*>(&features[i * dimension]), sizeof(double) * dimension);
      test(static_cast<size_t>(in.gcount()) == sizeof(double) * dimension, "Error reading features");

      in.read(reinterpret_cast<char*>(&weight[i]), sizeof(double));
      test(in.gcount() == sizeof(double), "Error reading weight");
    }
  }

  void write_accumulator(std::ostream& out, uint32_t size, uint32_t dimension,
                         std::vector<size_t> const& refs,
                         std::vector<double> const& features,
                         std::vector<double> const& weight) {
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (size_t i = 0ul; i < refs.size(); i++) {
      if (refs[i] == 0ul) {
        continue;
      }
      out.write(reinterpret_cast<const char*>(&dimension),               sizeof(dimension));
      out.write(reinterpret_cast<const char*>(&features[i * dimension]), sizeof(double) * dimension);
      out.write(reinterpret_cast<const char*>(&weight[i]),               sizeof(double));
    }
  }
}

/*****************************************************************************/

const ParameterString MixtureModel::paramLoadMixturesFrom("load-mixtures-from", "");

const char     MixtureModel::magic[8] = {'M', 'I', 'X', 'S', 'E', 'T', 0, 0};
const uint32_t MixtureModel::version = 2u;

/*****************************************************************************/

MixtureModel::MixtureModel(Configuration const& config, size_t dimension, size_t num_mixtures,
                           VarianceModel var_model, bool max_approx)
            : dimension(dimension),
              var_model(var_model),
              max_approx_    (max_approx),
              mixtures_(num_mixtures) {
  // TODO: implement
}

/*****************************************************************************/

void MixtureModel::reset_accumulators() {
  // TODO: implement
}

/*****************************************************************************/

void MixtureModel::accumulate(ConstAlignmentIter alignment_begin, ConstAlignmentIter alignment_end,
                              FeatureIter        feature_begin,   FeatureIter        feature_end,
                              bool first_pass, bool max_approx) {
  // TODO: implement
}

/*****************************************************************************/

void MixtureModel::finalize() {
  // TODO: implement
}

/*****************************************************************************/

void MixtureModel::split(size_t min_obs) {
  // TODO: implement
}

/*****************************************************************************/

void MixtureModel::eliminate(double min_obs) {
  // TODO: implement
}

/*****************************************************************************/

size_t MixtureModel::num_densities() const {
  return mean_refs_.size() - std::count(mean_refs_.begin(), mean_refs_.end(), 0ul);
}

/*****************************************************************************/

// computes the score (probability in negative log space) for a feature vector
// given a mixture density
double MixtureModel::density_score(FeatureIter const& iter, StateIdx mixture_idx, DensityIdx density_idx) const {
  // TODO: implement
  return 0.0;
}

/*****************************************************************************/

// this function returns the density with the lowest score (=highest probability)
// for the given feature vector
std::pair<double, DensityIdx> MixtureModel::min_score(FeatureIter const& iter, StateIdx mixture_idx) const {
  // TODO: implement
  return std::make_pair(0.0, 0u);
}

/*****************************************************************************/

// compute the 'full' score of a feature vector for a given mixture. The weights
// of each density are stored in weights and should sum up to 1.0
double MixtureModel::sum_score(FeatureIter const& iter, StateIdx mixture_idx, std::vector<double>* weights) const {
  // TODO: implement
  return 0.0;
}

/*****************************************************************************/

void MixtureModel::prepare_sequence(FeatureIter const& start, FeatureIter const& end) {
}

/*****************************************************************************/

double MixtureModel::score(FeatureIter const& iter, StateIdx mixture_idx) const {
  if (max_approx_) {
    return min_score(iter, mixture_idx).first;
  }
  else {
    return sum_score(iter, mixture_idx, NULL);
  }
}

/*****************************************************************************/

void MixtureModel::read(std::istream& in) {
  char     magic_test[8];
  uint32_t version_test;
  uint32_t dimension_test;

  in.read(magic_test, sizeof(magic_test));
  test(in.gcount() == sizeof(magic_test),             "Error reading magic header");
  test(std::string(magic) == std::string(magic_test), "Invalid magic header");

  in.read(reinterpret_cast<char*>(&version_test), sizeof(version_test));
  test(in.gcount() == sizeof(version_test),           "Error reading version");
  test(version == version_test,                       "Invalid version");

  in.read(reinterpret_cast<char*>(&dimension_test), sizeof(dimension_test));
  test(in.gcount() == sizeof(dimension_test),         "Error reading version");
  test(dimension == dimension_test,                   "Invalid version");

  read_accumulator(in, dimension, mean_refs_, mean_accumulators_, mean_weight_accumulators_);
  test(mean_refs_.size() < (1ul << 16),               "Too many means, mean indices are 16bit ints");
  mean_weights_.resize(mean_weight_accumulators_.size());
  means_.resize(mean_accumulators_.size());

  read_accumulator(in, dimension, var_refs_,  var_accumulators_,  var_weight_accumulators_);
  test(var_refs_.size() < (1ul << 16),                "Too many variances, var indices are 16bit ints");
  vars_.resize(var_accumulators_.size());
  norm_.resize(var_weight_accumulators_.size());

  uint32_t density_count;
  in.read(reinterpret_cast<char*>(&density_count), sizeof(density_count));
  test(in.gcount() == sizeof(density_count),          "Error reading density count");
  std::cerr << "Num densities: " << density_count << std::endl;

  std::vector<MixtureDensity> densities;
  densities.reserve(density_count);

  for (size_t i = 0ul; i < density_count; i++) {
    uint32_t mean_idx;
    in.read(reinterpret_cast<char*>(&mean_idx), sizeof(mean_idx));
    test(in.gcount() == sizeof(mean_idx),             "Error reading mean_idx");
    test(mean_idx < mean_refs_.size(),                "Invalid mean_idx");

    uint32_t var_idx;
    in.read(reinterpret_cast<char*>(&var_idx), sizeof(var_idx));
    test(in.gcount() == sizeof(var_idx),              "Error reading var_idx");
    test(var_idx < var_refs_.size(),                  "Invalid var_idx");

    mean_refs_[mean_idx] += 1ul;
    var_refs_[var_idx]   += 1ul;

    densities.push_back(MixtureDensity(mean_idx, var_idx));
  }

  uint32_t mixture_count;
  in.read(reinterpret_cast<char*>(&mixture_count), sizeof(mixture_count));
  test(in.gcount() == sizeof(mixture_count),          "Error reading mixture count");
  mixtures_.resize(mixture_count);
  std::cerr << "Num mixtures: " << mixture_count << std::endl;

  for (size_t m = 0ul; m < mixture_count; m++) {
    in.read(reinterpret_cast<char*>(&density_count), sizeof(density_count));
    test(in.gcount() == sizeof(density_count),        "Error reading density count for mixture");

    mixtures_[m].clear();
    mixtures_[m].reserve(density_count);
    for (size_t d = 0ul; d < density_count; d++) {
      uint32_t density_idx;
      in.read(reinterpret_cast<char*>(&density_idx), sizeof(density_idx));
      test(in.gcount() == sizeof(density_idx),        "Error reading density idx");
      test(density_idx < densities.size(),            "Invalid density idx");

      mixtures_[m].push_back(densities[density_idx]);

      double density_weight;
      const double expected_density_weight = mean_weight_accumulators_[densities[density_idx].mean_idx];
      in.read(reinterpret_cast<char*>(&density_weight), sizeof(density_weight));
      test(in.gcount() == sizeof(density_weight),     "Error reading density weight");
      test(density_weight == expected_density_weight, "Inconsistent density weight");
    }
  }

  finalize();
}

/*****************************************************************************/

void MixtureModel::write(std::ostream& out) const {
  const uint32_t dimension_u32 = dimension;
  out.write(magic, sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version),       sizeof(version));
  out.write(reinterpret_cast<const char*>(&dimension_u32), sizeof(dimension_u32));

  std::vector<size_t> mean_mapping(mean_refs_.size());
  std::vector<size_t> var_mapping (var_refs_.size());

  const uint32_t mean_count = build_mapping(mean_refs_, mean_mapping);
  const uint32_t var_count  = build_mapping(var_refs_,  var_mapping);

  write_accumulator(out, mean_count, dimension,
                    mean_refs_, mean_accumulators_, mean_weight_accumulators_);
  write_accumulator(out, var_count, dimension,
                    var_refs_,  var_accumulators_,  var_weight_accumulators_);

  uint32_t density_count = 0u;
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    density_count += mixtures_[m].size();
  }
  out.write(reinterpret_cast<const char*>(&density_count), sizeof(density_count));
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    for (size_t d = 0ul; d < mixtures_[m].size(); d++) {
      uint32_t mean_idx = mean_mapping[mixtures_[m][d].mean_idx];
      uint32_t var_idx  = var_mapping[mixtures_[m][d].var_idx];
      out.write(reinterpret_cast<const char*>(&mean_idx), sizeof(mean_idx));
      out.write(reinterpret_cast<const char*>(&var_idx),  sizeof(var_idx));
    }
  }

  const uint32_t mixture_count = mixtures_.size();
  out.write(reinterpret_cast<const char*>(&mixture_count), sizeof(mixture_count));
  uint32_t current_density = 0u;
  for (size_t m = 0ul; m < mixtures_.size(); m++) {
    density_count = mixtures_[m].size();
    out.write(reinterpret_cast<const char*>(&density_count), sizeof(density_count));
    for (size_t d = 0ul; d < mixtures_[m].size(); d++) {
      const size_t mean_idx = mixtures_[m][d].mean_idx;
      out.write(reinterpret_cast<const char*>(&current_density), sizeof(current_density));
      out.write(reinterpret_cast<const char*>(&mean_weight_accumulators_[mean_idx]), sizeof(double));
      current_density++;
    }
  }
}

/*****************************************************************************/
