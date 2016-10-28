/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "SignalAnalysis.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdint.h>
#include <string>
#include <vector>

#include "IO.hpp"
#include "Util.hpp"

/*****************************************************************************/

namespace {
  template<typename T>
  T add_sqr(T const& val, T const& acc) {
    return acc + val * val;
  }

  template<typename T>
  struct compute_stddev {
    T inv_num_obs_;

    compute_stddev(T const& inv_num_obs) : inv_num_obs_(inv_num_obs) {}

    T operator()(T const& mean, T const& sqrsum) {
      return std::sqrt(inv_num_obs_ * sqrsum - mean * mean);
    }
  };
}

/*****************************************************************************/

const ParameterBool   SignalAnalysis::paramEnergyMaxNorm  ("energy-max-norm", true);

const ParameterUInt64 SignalAnalysis::paramSampleRate     ("sample-rate",    8000ul);
const ParameterUInt64 SignalAnalysis::paramWindowShift    ("window-shift",     10ul);
const ParameterUInt64 SignalAnalysis::paramWindowSize     ("window-size",      25ul);
const ParameterUInt64 SignalAnalysis::paramDftLength      ("dft-length",     1024ul);
const ParameterUInt64 SignalAnalysis::paramNMelFilters    ("n-mel-filters",    15ul);
const ParameterUInt64 SignalAnalysis::paramNFeaturesInFile("n-features-file",  12ul);
const ParameterUInt64 SignalAnalysis::paramNFeaturesFirst ("n-features-first", 12ul);
const ParameterUInt64 SignalAnalysis::paramNFeaturesSecond("n-features-second", 1ul);
const ParameterUInt64 SignalAnalysis::paramDerivStep      ("deriv-step",        3ul);

/*****************************************************************************/

void SignalAnalysis::init_window(WindowType type) {
  switch (type) {
    case HAMMING:
      double denom = static_cast<double>(window_func_.size() - 1);
      for (size_t i = 0u; i < window_func_.size(); i++) {
        window_func_[i] = 0.54 - 0.46 * std::cos(2.0 * M_PI * static_cast<double>(i) / denom);
      }
      break;
  }
}

/*****************************************************************************/

void SignalAnalysis::process(std::string const& input_path, std::string const& output_path) {
  std::vector<short> samples = read_audio_file(input_path);

  /* open feature file */
  create_dir(output_path);
  std::ofstream features_out(output_path.c_str(), std::ios_base::out | std::ios_base::trunc);
  if (not features_out.good()) {
    std::cerr << "Error: cannot open '" << output_path << "'" << std::endl;
  }

  const size_t num_frames = (samples.size() + window_shift - 1ul) / window_shift;
  feature_seq_.resize(num_frames * n_features_total);

  pre_emphasis(samples);
  for (size_t start = 0u; start < samples.size(); start += window_shift) {
    apply_window(samples, start);
    fft(windowed_signal_, NULL, fft_real_, fft_imag_);
    abs_spectrum();
    calc_mel_filterbanks();
    std::transform(mel_filterbanks_.begin(), mel_filterbanks_.end(),
                   log_mel_filterbanks_.begin(), static_cast<double(*)(double)>(std::log));
    calc_cepstrum();
    std::copy(cepstrum_.begin(), cepstrum_.end(), feature_seq_.begin() + (start / window_shift) * n_features_total);
    write_floats_to_file(features_out, cepstrum_);
    num_obs_++;
  }
  
  add_deltas();

  // mean-variance normalization
  for (size_t t = 0ul; t < num_frames; t++) {
    std::transform(feature_seq_.begin() +  t        * n_features_total,
                   feature_seq_.begin() + (t + 1ul) * n_features_total,
                   mean_.begin(), mean_.begin(), std::plus<double>());
    std::transform(feature_seq_.begin() +  t        * n_features_total,
                   feature_seq_.begin() + (t + 1ul) * n_features_total,
                   sqrsum_.begin(), sqrsum_.begin(), add_sqr<double>);
  }

  if (energy_max_norm) {
    energy_max_normalization();
  }
}

/*****************************************************************************/

void SignalAnalysis::pre_emphasis(std::vector<short>& samples) {
  const int short_min = static_cast<int>(std::numeric_limits<short>::min());
  const int short_max = static_cast<int>(std::numeric_limits<short>::max());

  int prevsample = samples[0];
  for (size_t i = 1u; i < samples.size(); i++) {
    int sample = samples[i];
    int diff = sample - prevsample;
    samples[i] = static_cast<short>(std::max(std::min(diff, short_max), short_min));
    prevsample = sample;
  }
}

/*****************************************************************************/

void SignalAnalysis::apply_window(std::vector<short> const& samples, size_t start) {
  size_t size = std::min(samples.size() - start, window_func_.size());
  std::transform(window_func_.begin(), window_func_.begin() + size, samples.begin() + start,
                 windowed_signal_.begin(), std::multiplies<double>());

  /* zero padding */
  std::fill(windowed_signal_.begin() + size, windowed_signal_.end(), 0.0);
}

/*****************************************************************************/

void SignalAnalysis::fft(std::vector<double> const& signal_real, std::vector<double> const* signal_imag,
                         std::vector<double>&       fft_real,    std::vector<double>&       fft_imag,
                         bool inverse) {
  // for a detailed description of the FFT see:
  // http://www.engineeringproductivitytools.com/stuff/T0001/index.html

  static bool first_call = true;
  static std::vector<double> cos(dft_length / 2);
  static std::vector<double> sin(dft_length / 2);
  const  double phi  = M_PI / static_cast<double>(dft_length / 2);

  // precompute sine/cosine values once
  if (first_call) {
    for(size_t i = 0u; i < dft_length / 2; i++) {
      cos[i] =  std::cos(i * phi);
      sin[i] = -std::sin(i * phi);
    }
    first_call = false;
  }

  // initialize fft_real
  std::transform(signal_real.begin(), signal_real.end(), fft_real.begin(),
                 std::bind1st(std::multiplies<double>(), 1.0 / std::sqrt(static_cast<double>(dft_length))));

  // fft_imag is an optional parameter, if it is NULL we assume it consists of only zeros
  if (signal_imag == NULL) {
    std::fill(fft_imag.begin(), fft_imag.end(), 0.0);
  }
  else {
    // in case we want to compute the inverse fft we also need to conjugate the signal
    double conjugate = inverse ? -1.0 : 1.0;
    std::transform(signal_imag->begin(), signal_imag->end(), fft_imag.begin(),
                   std::bind1st(std::multiplies<double>(), conjugate / std::sqrt(static_cast<double>(dft_length))));
  }

  // perform bit-reverse permutation
  // the element at index b = b_log(dft_length) ... b_0
  // is placed at position b' = b_0 ... b_log(dft_length)
  // where b_i is the i'th bit of b
  size_t N2 = dft_length >> 1;
  size_t j = 0u;
  for (size_t i = 1u; i < dft_length; i++) {
    size_t m = N2;
    while (m <= j) {
      j -= m;
      m >>= 1;
    }
    j += m;
    if (j > i) {
      std::swap(fft_real[i], fft_real[j]);
      std::swap(fft_imag[i], fft_imag[j]);
    }
  }
  
  // perform DIF FFT
  for (size_t m = 1; m < dft_length; m <<= 1) {
    size_t Mm = m << 1;

    for (size_t k = 0; k < m; k++) {
      double re_w = cos[k * N2];
      double im_w = sin[k * N2];

      for(size_t i = k; i < dft_length; i += Mm) {
        j = i + m;
        double tmp1 = re_w * fft_real[j] - im_w * fft_imag[j];
        double tmp2 = im_w * fft_real[j] + re_w * fft_imag[j];

        fft_real[j]  = fft_real[i] - tmp1;
        fft_imag[j]  = fft_imag[i]  - tmp2;
        fft_real[i] += tmp1;
        fft_imag[i] += tmp2;
      }
    }

    N2 >>= 1;
  }
}

/*****************************************************************************/

void SignalAnalysis::abs_spectrum() {
  size_t size = std::min(std::min(fft_real_.size(), fft_imag_.size()), spectrum_.size());
  std::transform(fft_real_.begin(),
                 fft_real_.begin() + size,
                 fft_imag_.begin(),
                 spectrum_.begin(),
                 hypot);
}

/*****************************************************************************/

void SignalAnalysis::calc_mel_filterbanks() {
  // TODO: implement
}

/*****************************************************************************/

void SignalAnalysis::calc_cepstrum() {
  // TODO: implement
}

/*****************************************************************************/

void SignalAnalysis::add_deltas() {
  const size_t num_frames = feature_seq_.size() / n_features_total;
  for (size_t t = 0ul; t < num_frames; t++) {
    std::transform(feature_seq_.begin() +  std::max(t, deriv_step)               * n_features_total,
                   feature_seq_.begin() +  std::max(t, deriv_step)               * n_features_total + n_features_first,
                   feature_seq_.begin() + (std::max(t, deriv_step) - deriv_step) * n_features_total,
                   feature_seq_.begin() +  t                                     * n_features_total + n_features_in_file,
                   std::minus<float>());
  }
  for (size_t t = 0ul; t < num_frames; t++) {
    std::transform(feature_seq_.begin() + (std::min(t, num_frames - 1 - deriv_step) + deriv_step) * n_features_total + n_features_in_file,
                   feature_seq_.begin() + (std::min(t, num_frames - 1 - deriv_step) + deriv_step) * n_features_total + n_features_in_file + n_features_second,
                   feature_seq_.begin() +  t                                                      * n_features_total + n_features_in_file,
                   feature_seq_.begin() +  t                                                      * n_features_total + n_features_in_file + n_features_first,
                   std::minus<float>());
  }
}

/*****************************************************************************/

void SignalAnalysis::energy_max_normalization() {
  const size_t num_frames = feature_seq_.size() / n_features_total;
  float max_energy = -std::numeric_limits<float>::infinity();
  for (size_t t = 0ul; t < num_frames; t++) {
    max_energy = std::max(max_energy, feature_seq_[t * n_features_total]);
  }
  for (size_t t = 0ul; t < num_frames; t++) {
    feature_seq_[t * n_features_total] -= max_energy;
  }
}

/*****************************************************************************/

void SignalAnalysis::compute_normalization() {
  const double inv_num_obs = 1.0 / static_cast<double>(num_obs_);
  std::transform(mean_.begin(), mean_.end(), mean_.begin(),
                 std::bind1st(std::multiplies<double>(), inv_num_obs));
  std::transform(mean_.begin(), mean_.end(), sqrsum_.begin(),
                 stddev_.begin(), compute_stddev<double>(inv_num_obs));
  apply_mean_var_normalization_ = true;
}

/*****************************************************************************/

void SignalAnalysis::write_normalization_file(std::ostream& out) const {
  write_binary_blob(out, mean_);
  write_binary_blob(out, stddev_);
}

/*****************************************************************************/

void SignalAnalysis::read_normalization_file(std::istream& in) {
  read_binary_blob(in, mean_);
  read_binary_blob(in, stddev_);
  apply_mean_var_normalization_ = true;
}

/*****************************************************************************/

void SignalAnalysis::process_features(std::vector<float>& features) {
  const size_t num_frames = features.size() / n_features_in_file;
  feature_seq_.resize(num_frames * n_features_total);

  for (size_t f = 0ul; f < num_frames; f++) {
    std::copy(features.begin()     +  f        * n_features_in_file,
              features.begin()     + (f + 1ul) * n_features_in_file,
              feature_seq_.begin() +  f        * n_features_total);
  }
  add_deltas();
  if (apply_mean_var_normalization_) {
    for (std::vector<float>::iterator it = feature_seq_.begin(); it < feature_seq_.end(); it += n_features_total) {
      std::transform(it, it + n_features_total, mean_.begin(),   it, std::minus<double>());
      std::transform(it, it + n_features_total, stddev_.begin(), it, std::divides<double>());
    }
  }
  if (energy_max_norm) {
    energy_max_normalization();
  }
  std::swap(features, feature_seq_);
}

/*****************************************************************************/
