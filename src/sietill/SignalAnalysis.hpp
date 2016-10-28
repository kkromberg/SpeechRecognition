/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __SIGNAL_ANALYSIS_H__
#define __SIGNAL_ANALYSIS_H__

#include <string>

#include "Config.hpp"

class SignalAnalysis {
public:
  enum WindowType {
    HAMMING
  };

  static const ParameterBool   paramEnergyMaxNorm;

  static const ParameterUInt64 paramSampleRate;
  static const ParameterUInt64 paramWindowShift;
  static const ParameterUInt64 paramWindowSize;
  static const ParameterUInt64 paramDftLength;
  static const ParameterUInt64 paramNMelFilters;
  static const ParameterUInt64 paramNFeaturesInFile;
  static const ParameterUInt64 paramNFeaturesFirst;
  static const ParameterUInt64 paramNFeaturesSecond;
  static const ParameterUInt64 paramDerivStep;

  const bool energy_max_norm;

  const size_t sample_rate;
  const size_t window_shift;
  const size_t window_size;
  const size_t dft_length;
  const size_t n_mel_filters;
  const size_t n_features_in_file;
  const size_t n_features_first;
  const size_t n_features_second;
  const size_t n_features_total;
  const size_t deriv_step;

  SignalAnalysis(Configuration const& config) : energy_max_norm   (paramEnergyMaxNorm(config)),
                                                sample_rate       (paramSampleRate(config)),
                                                window_shift      (paramWindowShift(config) * sample_rate / 1000ul),
                                                window_size       (paramWindowSize(config)  * sample_rate / 1000ul),
                                                dft_length        (paramDftLength(config)),
                                                n_mel_filters     (paramNMelFilters(config)),
                                                n_features_in_file(paramNFeaturesInFile(config)),
                                                n_features_first  (paramNFeaturesFirst(config)),
                                                n_features_second (paramNFeaturesSecond(config)),
                                                n_features_total  (n_features_in_file + n_features_first + n_features_second),
                                                deriv_step        (paramDerivStep(config)),
                                                apply_mean_var_normalization_(false),
                                                window_func_        (window_size),
                                                windowed_signal_    (dft_length),
                                                fft_real_           (dft_length),
                                                fft_imag_           (dft_length),
                                                spectrum_           (dft_length / 2 + 1),
                                                mel_filterbanks_    (n_mel_filters),
                                                log_mel_filterbanks_(n_mel_filters),
                                                cepstrum_           (n_features_in_file),
                                                num_obs_            (0u),
                                                mean_               (n_features_total),
                                                stddev_             (n_features_total),
                                                sqrsum_             (n_features_total) {
    init_window(HAMMING);
    std::fill(  mean_.begin(),   mean_.end(), 0.0);
    std::fill(sqrsum_.begin(), sqrsum_.end(), 0.0);
  }

  ~SignalAnalysis() {}

  void init_window(WindowType type);
  void process(std::string const& input_path, std::string const& output_path);
  void pre_emphasis(std::vector<short>& samples);
  void apply_window(std::vector<short> const& samples, size_t start);
  void fft(std::vector<double> const& signal_real, std::vector<double> const* signal_imag,
           std::vector<double>&       fft_real,    std::vector<double>&       fft_imag,
           bool inverse=false);
  void abs_spectrum();
  void calc_mel_filterbanks();
  void calc_cepstrum();
  void add_deltas();
  void energy_max_normalization();

  void compute_normalization();
  void write_normalization_file(std::ostream& out) const;
  void read_normalization_file(std::istream& in);

  void process_features(std::vector<float>& features);
private:
  bool apply_mean_var_normalization_;

  std::vector<double> window_func_;

  std::vector<double> windowed_signal_;
  std::vector<double> fft_real_;
  std::vector<double> fft_imag_;
  std::vector<double> spectrum_;
  std::vector<double> mel_filterbanks_;
  std::vector<double> log_mel_filterbanks_;
  std::vector<double> cepstrum_;
  std::vector<float>  feature_seq_;

  size_t              num_obs_;
  std::vector<double> mean_;
  std::vector<double> stddev_;
  std::vector<double> sqrsum_;
};

#endif /* __SIGNAL_ANALYSIS_H__ */
