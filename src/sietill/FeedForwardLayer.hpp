/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#ifndef __FEED_FORWARD_LAYER_HPP__
#define __FEED_FORWARD_LAYER_HPP__

#include "NetworkLayer.hpp"

class FeedForwardLayer : public NetworkLayer {
public:
  enum struct Nonlinearity {
    None,
    Sigmoid,
    Tanh,
    ReLU,
  };

  enum struct WeightDecay {
    None,
    L1,
    L2
  };

  static const ParameterString paramNonlinearity;
  static const ParameterString paramWeightDecay;
  static const ParameterDouble paramWeightDecayFactor;

  FeedForwardLayer(Configuration const& config);
  virtual ~FeedForwardLayer();

  virtual void init(bool input_error_needed);

  virtual void init_parameters(std::function<float()> const& generator);
  virtual void forward (std::valarray<float>& output,
                        std::gslice const& slice, std::vector<unsigned> const& mask) const;
  virtual void backward_start();
  virtual void backward(std::valarray<float>& output, std::valarray<float>& error,
                        std::gslice const& slice, std::vector<unsigned> const& mask);
protected:
  FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity);
  FeedForwardLayer(Configuration const& config, Nonlinearity nonlinearity,
  																	 WeightDecay weight_decay,
  																	 float weight_decay_factor);
private:
  const Nonlinearity nonlinearity_;
  WeightDecay        weight_decay_;
  float              weight_decay_factor_;
};

#endif /* __FEED_FORWARD_LAYER_HPP__ */
