/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Config.hpp"
#include "Corpus.hpp"
#include "IO.hpp"
#include "NeuralNetwork.hpp"
#include "NNTraining.hpp"
#include "Recognizer.hpp"
#include "SignalAnalysis.hpp"
#include "Training.hpp"

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <config-file>" << std::endl;
    return EXIT_FAILURE;
  }

  const Configuration config(argv[1]);

  const ParameterString paramAction("action", "");
  const ParameterString paramFeaturePath("feature-path", "");
  const ParameterString paramNormalizationPath("normalization-path", "");
  const ParameterBool   paramMaxApprox("max-approx", true);
  const ParameterString paramFeatureScorer("feature-scorer", "gmm");
  const ParameterString paramAlignmentPath("alignment-path", "");


  std::string action(paramAction(config));
  std::string feature_path(paramFeaturePath(config));
  std::string normalization_path(paramNormalizationPath(config));
  bool        max_approx(paramMaxApprox(config));

  if (argc >= 3) {
    action = std::string(argv[2]);
  }

  Lexicon lexicon = build_sietill_lexicon();
  CorpusDescription corpus_description(config);
  corpus_description.read(lexicon);
  SignalAnalysis analyzer(config);

/*****************************************************************************/
  if (action == "extract-features") {
    const ParameterString paramAudioPath  ("audio-path",   "");
    const ParameterString paramAudioFormat("audio-format", "sph");

    std::string audio_path  (paramAudioPath  (config));
    std::string audio_format(paramAudioFormat(config));

    /* proceed over training/test samples and perform signal analysis */
    size_t i = 0ul;
    for (auto seg_iter = corpus_description.begin(); seg_iter != corpus_description.end(); ++seg_iter) {
      i++;
      std::cerr << "Processing (" << i << "): " << seg_iter->name << std::endl;
      analyzer.process(audio_path   + seg_iter->name + std::string(".") + audio_format,
                       feature_path + seg_iter->name + std::string(".mm2"));
    }
    if (normalization_path.size() > 0) {
      std::ofstream normalization_stream(normalization_path.c_str(), std::ios_base::out | std::ios_base::trunc);
      if (not normalization_stream.good()) {
        std::cerr << "Error: could not open normalization file" << std::endl;
        exit(EXIT_FAILURE);
      }
      analyzer.compute_normalization();
      analyzer.write_normalization_file(normalization_stream);
    }
  }
/*****************************************************************************/
  else if (action == "train" or action == "recognize") {
    if (normalization_path.size() > 0) {
      std::ifstream normalization_stream(normalization_path.c_str(), std::ios_base::in);
      if (not normalization_stream.good()) {
        std::cerr << "Error: could not open normalization file" << std::endl;
        exit(EXIT_FAILURE);
      }
      analyzer.read_normalization_file(normalization_stream);
    }

    const ParameterString paramPooling  ("pooling",   "");
    std::string poolingString  (paramPooling  (config));

    MixtureModel::VarianceModel pooling_method = MixtureModel::NO_POOLING;
    if (poolingString == "none") {
    	pooling_method = MixtureModel::NO_POOLING;
    } else if (poolingString == "mixture") {
    	pooling_method = MixtureModel::MIXTURE_POOLING;
    } else if (poolingString == "global") {
    	pooling_method = MixtureModel::GLOBAL_POOLING;
    } else {
    	std::cerr << "ERROR: The following GMM pooling option is not valid: " << poolingString << std::endl;
    }

    Corpus corpus;
    corpus.read(corpus_description, feature_path, analyzer);

    TdpModel tdp_model(config, lexicon.get_silence_automaton()[0ul]);

    if (action == "train") {
      MixtureModel mixtures(config, analyzer.n_features_total, lexicon.num_states(), pooling_method, max_approx);

      Trainer trainer(config, lexicon, mixtures, tdp_model, max_approx);
      trainer.train(corpus);
    }
    else { // action == "recognize"
      std::string feature_scorer = paramFeatureScorer(config);
      FeatureScorer* fs = nullptr;

      if (feature_scorer == "gmm") {
        fs = new MixtureModel(config, analyzer.n_features_total, lexicon.num_states(), pooling_method, max_approx);
      }
      else if (feature_scorer == "nn") {
        const ParameterUInt paramContextFrames("context-frames", 0);
        size_t context_frames = paramContextFrames(config);
        fs = new NeuralNetwork(config, analyzer.n_features_total * (2ul * context_frames + 1ul), 1ul, corpus.get_max_seq_length(), lexicon.num_states());
        dynamic_cast<NeuralNetwork*>(fs)->load_prior();
      }
      else {
        std::cerr << "unknown feature scorer: " << feature_scorer << std::endl;
        exit(EXIT_FAILURE);
      }

      Recognizer recognizer(config, lexicon, *fs, tdp_model);
      recognizer.recognize(corpus);
    }
  }
/*****************************************************************************/
  else if (action == "train-nn" or action == "compute-prior" or action == "plot-activations") {
    const ParameterUInt paramBatchSize("batch-size", 32u);
    const unsigned batch_size(paramBatchSize(config));

    Corpus corpus;
    corpus.read(corpus_description, feature_path, analyzer);

    MiniBatchBuilder mini_batch_builder(config, corpus, batch_size, lexicon.num_states(), lexicon.get_silence_automaton()[0]);
    NeuralNetwork    nn(config, mini_batch_builder.feature_size(), batch_size, corpus.get_max_seq_length(), lexicon.num_states());

    if (action == "train-nn") {
      NnTrainer nn_trainer(config, mini_batch_builder, nn);
      nn_trainer.train();
    }
    else if (action == "plot-activations") {
    	std::vector<std::string> layer_names = {"hidden-layer1", "hidden-layer2", "output-layer"};

    	std::valarray<float> targets = std::valarray<float>(1, 0.0);
      mini_batch_builder.build_batch(0, false,
                                      nn.get_feature_buffer(),
                                      nn.get_feature_buffer_slice(),
                                      targets,
                                      nn.get_batch_mask());
      nn.forward();

      for (size_t i = 0; i < layer_names.size(); i++) {
      	NetworkLayer *layer = nn.get_network_layer(layer_names[i]);
      	assert(layer != nullptr);

      	std::valarray<float> inputs = layer->get_input_buffer();
      	size_t feature_size = layer->get_feature_size();

    		std::vector<double> feature_vector(feature_size, 0.0);
    		std::ofstream output("../activation-plotting/activations/" + layer_names[i] + ".activations", std::ios_base::out | std::ios_base::trunc);
      	for (size_t time_idx = 0; time_idx < layer->get_max_seq_length_(); time_idx++) {
        	for (size_t input_idx = 0; input_idx < feature_size; input_idx++) {
        		feature_vector[input_idx] = inputs[time_idx * feature_size + time_idx];
        	}
      		write_floats_to_file(output, feature_vector);
      	}
      	output.close();
      }

    } else { // action == "compute-prior"
      const ParameterString paramPriorFile("prior-file", "");
      std::string prior_file = paramPriorFile(config);
      const ParameterBool computePriorFromAlignment("compute-prior-from-alignment", "");
      bool compute_prior_from_alignment = computePriorFromAlignment(config);

      std::ofstream out(prior_file, std::ios::out | std::ios::trunc);
      if (not out.good()) {
        std::cerr << "Could not open prior-file: " << prior_file << std::endl;
        std::abort();
      }

      if (compute_prior_from_alignment) {
        // compute prior using alignment
        // TODO: implement
        std::vector<float> state_frequencies(lexicon.num_states());
        size_t sum = 0;
        Alignment alignment  = mini_batch_builder.get_alignment();
        size_t num_max_align = mini_batch_builder.get_max_align();
        ConstAlignmentIter align_begin(&*alignment.begin(), num_max_align);
        ConstAlignmentIter align_end(&*alignment.end(), num_max_align);

        for (ConstAlignmentIter align_iter = align_begin; align_iter != align_end; align_iter++) {
        	sum += 1;
        	state_frequencies[(*align_iter)->state] += 1;
        }

        //std::cerr << "LENGHT: " << state_frequencies.size() << std::endl;
        //std::cerr << "SUM: "		<< sum											<< std::endl;
        for (size_t i = 0; i < state_frequencies.size(); i++) {
        	//std::cerr << "I : " << state_frequencies[i]/sum << std::endl;
        	out << state_frequencies[i]/sum << " ";
        }
      }
      else {	// compute prior using models
        // load weights
      	const ParameterString modelPath("model-path", "");
      	std::string model_path = modelPath(config);
        //std::string model_path = "./models/1/";
        nn.load(model_path);

        //std::cerr << "FEATURE: " << nn.get_feature_buffer().sum() << std::endl;
        // do forward step
        nn.forward();
        std::valarray<float> score = nn.get_score_buffer();
        std::cerr << "OUTPUT SIZE: " << score.size() << std::endl;
        for (size_t i = 0; i < lexicon.num_states(); i++) {
        	out << score[i] << " ";
        	std::cerr << score[i] << std::endl;
        }
      }
      out.close();
    }
  }
/*****************************************************************************/
  else {
    std::cerr << "Error: unknown action " << action << std::endl;
    exit(EXIT_FAILURE);
  }
/*****************************************************************************/

  return EXIT_SUCCESS;
}

