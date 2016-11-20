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

  template<typename T>
  struct dump_vector {
    dump_vector(std::ostream& stream, std::vector<T> const& vector) {
      for (auto it : vector ) {
        stream << it << " ";
      }
      stream << "\n";
    }
  };

  template<typename T>
  struct minus_power {
    T operator()(T const& x, T const& y) {
      return x - y * y;
    }
  };

  template<typename T>
  struct scale_add {
    T scale_;

    scale_add(T const& scale) : scale_(scale) {}

    T operator()(T const& x, T const& y) {
      return x + scale_ * y;
    }
  };

  template<typename T>
  struct scale_add_square {
    T scale_;

    scale_add_square(T const& scale) : scale_(scale) {}

    T operator()(T const& x, T const& y) {
      return x + scale_ * y * y;
    }
  };


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
const ParameterString MixtureModel::paramVerbosity			 ("verbosity", "");

const char     MixtureModel::magic[8] = {'M', 'I', 'X', 'S', 'E', 'T', 0, 0};
const uint32_t MixtureModel::version = 2u;

/*****************************************************************************/

MixtureModel::MixtureModel(Configuration const& config, size_t dimension, size_t num_mixtures,
                           VarianceModel var_model, bool max_approx)
            : dimension(dimension),
              var_model(var_model),
              max_approx_(max_approx),
              mixtures_(num_mixtures),
              verbosity_(get_verbosity_from_string(paramVerbosity(config))) {
  // TODO: implement

	for (size_t mixture_counter = 0; mixture_counter < num_mixtures; mixture_counter++) {
		Mixture mixture;

		// create one mixture density for each mixture
		if (var_model != GLOBAL_POOLING) {
		  MixtureDensity mixture_density = create_mixture_density(mean_refs_.size(), var_refs_.size());
	    mixture.push_back(mixture_density);
		} else {
	    MixtureDensity mixture_density = create_mixture_density(mean_refs_.size(), 0);
	    mixture.push_back(mixture_density);
		}

		mixtures_[mixture_counter] = mixture;
	}
}

/*****************************************************************************/

MixtureDensity MixtureModel::create_mixture_density(DensityIdx mean_index, DensityIdx var_index) {
  //test(mean_refs_.size() > mean_index, "ERROR: create_mixture mixture index already exists");

  bool reuse_variance_containers = false;
  if (var_index < var_refs_.size()) {
    reuse_variance_containers = true;
  }

  mean_refs_.push_back(1);
  mean_weights_.push_back(0.0);
  mean_weight_accumulators_.push_back(0.0);
  for (size_t feature_counter = 0; feature_counter < dimension; feature_counter++) {
    means_.push_back(0.0);
    mean_accumulators_.push_back(0.0);
  }

  if (!reuse_variance_containers) {
    norm_.push_back(0.0);
    var_weight_accumulators_.push_back(0.0);
    var_refs_.push_back(1);
    for (size_t feature_counter = 0; feature_counter < dimension; feature_counter++) {
      vars_.push_back(0.0);
      var_accumulators_.push_back(1e-6);
    }
  }

  MixtureDensity density = MixtureDensity(mean_index, var_index);
  return density;
}

/*****************************************************************************/

void MixtureModel::reset_accumulators() {
  // TODO: implement
	// reset accumulation vectors
	std::fill(mean_accumulators_.begin(), mean_accumulators_.end(), 0.0);
	std::fill(mean_weight_accumulators_.begin(), mean_weight_accumulators_.end(), 0.0);

	std::fill(var_accumulators_.begin(), var_accumulators_.end(), 1e-6); // Avoid numerical instability
	std::fill(var_weight_accumulators_.begin(), var_weight_accumulators_.end(), 0.0);
}

/******************************************************************************/
/**
 * Implements the following equation E[X^2] - E[X]^2
 */
void MixtureModel::calculate_variance(DensityIdx var_idx, std::vector<double>::iterator means_iterator_begin) {
  std::transform(var_accumulators_.begin() + var_idx * dimension,
                 var_accumulators_.begin() + var_idx * dimension + dimension,
                 vars_.begin() + var_idx * dimension,
                 std::bind2nd(std::divides<double>(), var_weight_accumulators_[var_idx]));

  std::transform(vars_.begin()  + var_idx  * dimension,
                 vars_.begin()  + var_idx  * dimension + dimension,
                 means_iterator_begin,
                 vars_.begin()  + var_idx  * dimension,
                 minus_power<double>());

  norm_[var_idx] = std::accumulate(vars_.begin() + var_idx * dimension,
      vars_.begin() + var_idx * dimension + dimension,
      dimension * log(2*M_PI),
      [](double const& x, double const& y) {
          return x + log(y);
      });
  norm_[var_idx]/= 2;
}
/*****************************************************************************/

void MixtureModel::accumulate(ConstAlignmentIter alignment_begin, ConstAlignmentIter alignment_end,
                              FeatureIter        feature_begin,   FeatureIter        feature_end,
                              bool first_pass, bool max_approx) {
	if (verbosity_ > noLog) {
	  std::cout<<"Estimation step"<<std::endl;
	}

	reset_accumulators();

  ConstAlignmentIter alignment_iterator = alignment_begin;
  for (FeatureIter feature_iterator = feature_begin;
       feature_iterator != feature_end;
       feature_iterator++, alignment_iterator++) {

    StateIdx m = (*alignment_iterator)->state;
    std::vector<double> membership_probabilities = std::vector<double>(mixtures_[m].size(), 0.0);

    DensityIdx max_approx_density = 0;
    if (max_approx && !first_pass) {
      max_approx_density = min_score(feature_iterator, m).second;
    }

    for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {
      if(first_pass) {
        //at the first pass we have only one density
        membership_probabilities[0] = 1.0;
      } else if (max_approx && d == max_approx_density) {
        // set the membership probability to 1 for the max_approx case
        membership_probabilities[d] = 1.0;
      } else if (!max_approx) {
        membership_probabilities[d] = mean_weights_[mixtures_[m][d].mean_idx]
                                    * exp(-1 * density_score(feature_iterator, m , d));
      }
    }

    if (verbosity_ > noLog) {
      //dump_vector<float>(std::cerr, membership_probabilities);
    }

    if (!max_approx) {
      // Normalize the membership probabilities
      double sum_probability = std::accumulate(membership_probabilities.begin(),
                                               membership_probabilities.end()  , 0.0);
      std::transform(membership_probabilities.begin(), membership_probabilities.end(),
                     membership_probabilities.begin(), std::bind2nd(std::divides<double>(), sum_probability));

      if (verbosity_ > noLog) {
        std::cout << "sum_probability: " << sum_probability << std::endl;
      }
    }

    if (verbosity_ > noLog) {
      //dump_vector<double>(std::cout, membership_probabilities);
    }

    for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {

      // Skip entries which can be ignored
      if (membership_probabilities[d] == 0.0) {
        continue;
      }

      DensityIdx mean_idx = mixtures_[m][d].mean_idx;
      DensityIdx var_idx  = mixtures_[m][d].var_idx;

      mean_weight_accumulators_[mean_idx] += membership_probabilities[d];
      var_weight_accumulators_ [var_idx ] += membership_probabilities[d];

      std::transform(mean_accumulators_.begin() + mean_idx * dimension,
                     mean_accumulators_.begin() + mean_idx * dimension + dimension,
                     *feature_iterator,
                     mean_accumulators_.begin() + mean_idx * dimension,
                     scale_add<double>(membership_probabilities[d]));

      std::transform(var_accumulators_.begin() + var_idx * dimension,
                     var_accumulators_.begin() + var_idx * dimension + dimension,
                     *feature_iterator,
                     var_accumulators_.begin() + var_idx * dimension,
                     scale_add_square<double>(membership_probabilities[d]));

      //dump_vector<double>(std::cout, mean_accumulators_);
      //dump_vector<double>(std::cout, var_accumulators_);
    }
  }
  if (verbosity_ > noLog) {
    std::cout << "mean_accumulators_" << std::endl;
    dump_vector<double>(std::cout, mean_accumulators_);
    std::cout << "var_accumulators_" << std::endl;
    dump_vector<double>(std::cout, var_accumulators_);
    std::cout << "mean_weight_accumulators_" << std::endl;
    dump_vector<double>(std::cout, mean_weight_accumulators_);
    std::cout << "var_weight_accumulators_" << std::endl;
    dump_vector<double>(std::cout, var_weight_accumulators_);
  }
}

void MixtureModel::finalize() {
  if (verbosity_ > noLog) {
    std::cout<<"Maximization step"<<std::endl;
  }

  double total_observations = 0.0;
  for (StateIdx m = 0; m < mixtures_.size(); m++) {

    double total_mixture_observations = 0.0;
    for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {
      DensityIdx mean_idx = mixtures_[m][d].mean_idx;
      DensityIdx var_idx  = mixtures_[m][d].var_idx;

      // accumulate counts of the mixture (for the density weights)
      total_mixture_observations += mean_weight_accumulators_[mean_idx];

      // calculate the means of the density
      std::transform(mean_accumulators_.begin() + mean_idx * dimension,
                     mean_accumulators_.begin() + mean_idx * dimension + dimension,
                     means_.begin() + mean_idx * dimension,
                     std::bind2nd(std::divides<double>(), mean_weight_accumulators_[mean_idx]));

      // calculate the variance for the current density in the mixture
      if (var_model == NO_POOLING) {
        calculate_variance(var_idx, means_.begin() + mean_idx * dimension);
      }

    }

    // Calculate the density weights
    for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {
      DensityIdx mean_idx = mixtures_[m][d].mean_idx;
      mean_weights_[mean_idx] = mean_weight_accumulators_[mean_idx] / total_mixture_observations;
    }


    if (var_model == MIXTURE_POOLING) {
      // calculate the variance of the mixture
      // we collect the means of the whole mixture and use it to calculate the variance afterwards
      std::vector<double> mixture_mean  = std::vector<double>(dimension, 0.0);
      for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {
        std::transform(mixture_mean.begin(), mixture_mean.end(),
                       mean_accumulators_.begin() + mixtures_[m][d].mean_idx * dimension,
                       mixture_mean.begin(),
                       std::plus<double>());
      }

      // divide each dimension of the accumulated means by the number of points in the mixture
      std::transform(mixture_mean.begin(), mixture_mean.end(),
                     mixture_mean.begin(), std::bind2nd(std::divides<double>(), total_mixture_observations));

      // calculate the variance with the updated means
      DensityIdx var_idx  = mixtures_[m][0].var_idx;
      calculate_variance(var_idx, mixture_mean.begin());
    }

    total_observations += total_mixture_observations;
  }

  if (var_model == GLOBAL_POOLING) {
    // calculate the variance of the mixture
    // we collect the means of the whole mixture and use it to calculate the variance afterwards
    std::vector<double> mixture_mean  = std::vector<double>(dimension, 0.0);
    for (StateIdx m = 0; m < mixtures_.size(); m++) {
      for (DensityIdx d = 0; d < mixtures_[m].size(); d++) {
        std::transform(mixture_mean.begin(), mixture_mean.end(),
                       mean_accumulators_.begin() + mixtures_[m][d].mean_idx * dimension,
                       mixture_mean.begin(),
                       std::plus<double>());
      }
    }

    // divide each dimension of the accumulated means by the number of points of the whole dataset
    std::transform(mixture_mean.begin(), mixture_mean.end(),
                   mixture_mean.begin(), std::bind2nd(std::divides<double>(), total_observations));

    // calculate the variance with the updated means
    DensityIdx var_idx  = mixtures_[0][0].var_idx;
    calculate_variance(var_idx, mixture_mean.begin());
  }

  if (verbosity_ > noLog) {
    std::cout << "means_" << std::endl;
    dump_vector<double>(std::cout, means_);
    std::cout << "vars_" << std::endl;
    dump_vector<double>(std::cout, vars_);
    std::cout << "mean_weights_" << std::endl;
    dump_vector<double>(std::cout, mean_weights_);
  }
}

/*****************************************************************************/

void MixtureModel::split(size_t min_obs) {
  // TODO: implement
	unsigned int mean_ref, var_ref;
	for (unsigned int mixture = 0; mixture < mixtures_.size(); mixture++) {// loop mixtures

		for (int density = mixtures_[mixture].size() - 1; density >= 0; density--) {		// loop densities
			mean_ref = mixtures_[mixture][density].mean_idx * dimension;
			var_ref  = mixtures_[mixture][density].var_idx * dimension;

			// if accumulated weight > min_obs we have to split density
			if (mean_weight_accumulators_[mixtures_[mixture][density].mean_idx] >= min_obs) { // or weight acc?

			  MixtureDensity new_md = MixtureDensity(0, 0);

        // create a new density with according indices
			  if (var_model == NO_POOLING) {
          new_md = create_mixture_density(mean_refs_.size(), var_refs_.size());
			  } else {
          new_md = create_mixture_density(mean_refs_.size(), mixtures_[mixture][density].var_idx);
			  }

        update_split_densities(mixtures_[mixture][density], new_md);

        // put new density into current mixture
        mixtures_[mixture].push_back(new_md);
			}
		}
	}
}

void MixtureModel::update_split_densities(MixtureDensity& md_original, MixtureDensity& md_split) {

  DensityIdx mean_idx_original = md_original.mean_idx;
  DensityIdx var_idx_original  = md_original.var_idx;
  DensityIdx mean_idx_split    = md_split.mean_idx;
  DensityIdx var_idx_split     = md_split.var_idx;

  // copy mixture weights from the reference density
  mean_weights_[mean_idx_split]  = mean_weights_[mean_idx_original];

  for (unsigned int dim = 0; dim < dimension; dim++) {
    // calculate the term to shift the means by
    double mean_shift = sqrt(vars_[var_idx_original * dimension + dim]);

    // calculate the new means
    double mean_plus  = means_[mean_idx_original * dimension + dim] + mean_shift;
    double mean_minus = means_[mean_idx_original * dimension + dim] - mean_shift;

    // update the means
    means_[mean_idx_original * dimension + dim] = mean_plus;
    means_[mean_idx_split    * dimension + dim] = mean_minus;
  }

  if ( var_model == NO_POOLING ) {
    // copy the accumulators to calculate the variances / normalization terms
    var_weight_accumulators_[var_idx_split] = var_weight_accumulators_[var_idx_original];
    std::copy(var_accumulators_.begin() + dimension * var_idx_original,
              var_accumulators_.begin() + dimension * var_idx_original + dimension,
              var_accumulators_.begin() + dimension * var_idx_split);

    calculate_variance(var_idx_original, means_.begin() + mean_idx_original * dimension);
    calculate_variance(var_idx_split   , means_.begin() + mean_idx_split    * dimension);
  } else {
    // When using pooling, we can just copy the variances and normalization terms
    std::copy(vars_.begin() + dimension * var_idx_original,
              vars_.begin() + dimension * var_idx_original + dimension,
              vars_.begin() + dimension * var_idx_split);
     norm_[var_idx_split] = norm_[var_idx_original];
  }

}

/*****************************************************************************/

void MixtureModel::eliminate(double min_obs) {
  // TODO: implement
	unsigned int mean_ref, var_ref;

	//loop mixtures
	for (unsigned int mixture = 0; mixture < mixtures_.size(); mixture++) {
		// loop densities
		for (int density = mixtures_[mixture].size()-1; density >= 0; density--) {

			mean_ref = mixtures_[mixture][density].mean_idx;
			var_ref  = mixtures_[mixture][density].var_idx;

			// remove density if mean weight < min obs
			if (mean_weight_accumulators_[mean_ref] < min_obs) {
				mixtures_[mixture].erase(mixtures_[mixture].begin() + density);

				// set refs to 0
				mean_refs_[mean_ref] = 0ul;

				if (var_model == NO_POOLING) {
				  // when using pooling, we do not want to set any variance references to 0
				  var_refs_[var_ref]   = 0ul;
				}
			}
		}
	}
}

/*****************************************************************************/

size_t MixtureModel::num_densities() const {
  return mean_refs_.size() - std::count(mean_refs_.begin(), mean_refs_.end(), 0ul);
}

/*****************************************************************************/

// computes the score (probability in negative log space) for a feature vector
// given a mixture density
// Miguel: Here we do not need the membership probabilities of the density in the mixture
// 				 These are passed to the sum_score function, but not to this one.
double MixtureModel::density_score(FeatureIter const& iter, StateIdx mixture_idx, DensityIdx density_idx) const {
	double score = 0.0;

	// Positions of beginning of the mean / variance of the density in the flat array
	double mean_index     = mixtures_[mixture_idx][density_idx].mean_idx * dimension;
	double variance_index = mixtures_[mixture_idx][density_idx].var_idx  * dimension;

	//double variance_factor = log(pow(2 * M_PI, dimension));;
	double distance_factor    = 0.0;
	double distance_from_mean = 0.0;
	for (size_t feature_idx = 0; feature_idx < dimension; feature_idx++) {

		// update the mean term
		distance_from_mean = (*iter)[feature_idx] - means_[mean_index + feature_idx];
		distance_factor    += distance_from_mean * distance_from_mean / vars_[variance_index + feature_idx];

		if (verbosity_ > noLog) {
		  /*
			std::cout << "Current mean / variance " << means_[mean_index + feature_idx] << " "
                                              << vars_[variance_index + feature_idx] << " "
                                              << std::endl;
			std::cout << "Current distance_factor " << distance_factor << std::endl;
			*/

		}

		//variance_factor += log(vars_[variance_index + feature_idx]);
	}

	// apply operations on the end product of the terms and sum them
	score = norm_[mixtures_[mixture_idx][density_idx].var_idx] + distance_factor / 2;
  //score = (variance_factor + distance_factor) / 2;
	if (verbosity_ > noLog) {
/*
		std::cout << "Score of density: " << score
		          << " Probability:      " << exp(-1 * score)
							<< " Mixture idx:      " << mixture_idx
							<< " Density idx:      " << density_idx
							<< std::endl;
*/
	}

  return score;
}

/*****************************************************************************/

// this function returns the density with the lowest score (=highest probability)
// for the given feature vector
std::pair<double, DensityIdx> MixtureModel::min_score(FeatureIter const& iter, StateIdx mixture_idx) const {
  size_t     n_densities = mixtures_[mixture_idx].size();
  DensityIdx min_idx = 0;
  double     min_score = 1e10;
  double     new_score = 0.0;

  for (size_t density_idx = 0; density_idx < n_densities; density_idx++) {
    new_score = density_score(iter, mixture_idx, density_idx);

    // update density
    if (new_score < min_score) {
      min_idx   = density_idx;
      min_score = new_score;
    }
  }

  return std::make_pair(min_score, min_idx);
}

/*****************************************************************************/

// compute the 'full' score of a feature vector for a given mixture. The weights
// of each density are stored in weights and should sum up to 1.0
double MixtureModel::sum_score(FeatureIter const& iter, StateIdx mixture_idx, std::vector<double>* weights) const {

	assert(weights != NULL);
	if (verbosity_ > debugLog) {
		double sum = 0.0;
		for (size_t i = 0; i < weights->size(); i++) {
			sum += weights->at(i);
		}
		std::cerr << "Sum of weights in MixtureModel::sum_score " << sum << std::endl;
	}

  size_t     n_densities = mixtures_[mixture_idx].size();
  double     score = 0.0;

  for (size_t density_idx = 0; density_idx < n_densities; density_idx++) {
  	score += (*weights)[density_idx] * pow(M_E, -1 * density_score(iter, mixture_idx, density_idx));
  }

  return -log(score);
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
  	std::vector<double> weights;
  	size_t						  n_densities = mixtures_[mixture_idx].size();
  	size_t							weight_idx = 0;

  	for (size_t density_idx = 0; density_idx < n_densities; density_idx++) {
  		weight_idx = mixtures_[mixture_idx][density_idx].mean_idx;
  		weights.push_back(mean_weights_[weight_idx]);
  	}

    return sum_score(iter, mixture_idx, &weights);
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
