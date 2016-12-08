/*****************************************************************************/
/*                                                                           */
/*       COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen        */
/*                                                                           */
/*****************************************************************************/

#include "Config.hpp"

#include <exception>
#include <fstream>
#include <iostream>

#define RAPIDJSON_HAS_STDSTRING true

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

struct Configuration::Internal {
  std::shared_ptr<rapidjson::Document> parent;
  rapidjson::Value val;

  Internal(std::shared_ptr<rapidjson::Document> const& parent, rapidjson::Value const& src) : parent(parent), val(src, parent->GetAllocator()) {
  }
};

Configuration::Configuration() : internal_(new Internal(std::shared_ptr<rapidjson::Document>(nullptr), rapidjson::Value(rapidjson::Type::kObjectType))) {
}

Configuration::Configuration(std::unique_ptr<Internal>&& internal) : internal_(std::move(internal)) {
}

Configuration::Configuration(Configuration&& other) : internal_(std::move(other.internal_)) {
}
  
Configuration::Configuration(std::string const& path) {
  std::ifstream input(path, std::ios::in);
  rapidjson::IStreamWrapper input_wrapper(input);
  std::shared_ptr<rapidjson::Document> d(new rapidjson::Document());
  d->ParseStream(input_wrapper);
  if (d->HasParseError()) {
    std::cerr << "Error parsing config at " << d->GetErrorOffset() << ":" << rapidjson::GetParseError_En(d->GetParseError()) << std::endl;
    abort();
  }

  if (not d->IsObject()) {
    std::cerr << "Top level configuration is not an object" << std::endl;
    std::abort();
  }

  internal_.reset(new Internal{d, *d});
}

Configuration::~Configuration() {
}

bool Configuration::has_value(std::string const& name) const {
  return internal_->val.HasMember(name);
}

template<typename T>
bool Configuration::is_type(std::string const& name) const {
  return internal_->val[name.c_str()].Is<T>();
}

template<typename T>
T Configuration::get_value(std::string const& name) const {
  return internal_->val[name.c_str()].Get<T>();
}

bool Configuration::is_array(std::string const& name) const {
  return internal_->val[name.c_str()].IsArray();
}

std::vector<Configuration> Configuration::get_array(std::string const& name) const {
  std::vector<Configuration> result;
  for (auto iter = internal_->val[name.c_str()].Begin(); iter != internal_->val[name.c_str()].End(); ++iter) {
    result.emplace_back(std::unique_ptr<Internal>(new Internal(internal_->parent, *iter)));
  }
  return result;
}

std::vector<std::string> Configuration::get_string_array(std::string const& name) const {
  std::vector<std::string> result;
  for (auto iter = internal_->val[name.c_str()].Begin(); iter != internal_->val[name.c_str()].End(); ++iter) {
    result.emplace_back(iter->GetString(), iter->GetStringLength());
  }
  return result;
}

Configuration Configuration::sub_config(std::string const& name) const {
  return Configuration(std::unique_ptr<Internal>(new Internal(internal_->parent, internal_->val[name.c_str()])));
}

void Configuration::write_to_stream(std::ostream& out) const {
  rapidjson::OStreamWrapper osw(out);
  rapidjson::Writer<rapidjson::OStreamWrapper> writer(osw);
  this->internal_->parent->Accept(writer);
}

// -------------------- Parameter --------------------

template<typename T>
Parameter<T>::Parameter(std::string const& name, T const& default_value) : name_(name), default_value_(default_value) {
}

template<typename T>
Parameter<T>::~Parameter() {
}

template<typename T>
T Parameter<T>::operator()(Configuration const& config) const {
  if (config.has_value(name_)) {
    if (config.is_type<T>(name_)) {
      return config.get_value<T>(name_);
    }
    else {
      throw std::runtime_error(name_ + std::string(" has invalid type"));
    }
  }
  else {
    return default_value_;
  }
}

template class Parameter<bool>;
template class Parameter<int32_t>;
template class Parameter<uint32_t>;
template class Parameter<int64_t>;
template class Parameter<uint64_t>;
template class Parameter<float>;
template class Parameter<double>;
template class Parameter<std::string>;

// -------------------- Verbosity --------------------
Verbosity get_verbosity_from_string(std::string input) {
	if (input == "noLog") {
		return noLog;
	} else if (input == "informationLog") {
		return informationLog;
	} else if (input == "debugLog") {
		return debugLog;
	} else {
		throw std::runtime_error("Verbosity: " + input + std::string(" has invalid type"));
		return noLog;
	}
}
//neural network method
NNMethod get_method_from_string(std::string input) {
	if (input == "newBob") {
		return newBob;
	} else if (input == "no") {
		return no;
	}  else {
		throw std::runtime_error("NNMethod: " + input + std::string(" has invalid type"));
		return no;
	}
}




