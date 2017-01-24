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
#include <Modules.hh>
#include <Search/Module.hh>
#include <Search/WordConditionedTreeSearch.hh>
#include <Teaching/WordConditionedTreeSearch.hh>

using namespace Search;

Module_::Module_()
{
}


SearchAlgorithm* Module_::createRecognizer(SearchType type, const Core::Configuration &config) const
{
    SearchAlgorithm *recognizer = 0;
    switch (type) {
    case WordConditionedTreeSearchType:
	recognizer = new Search::WordConditionedTreeSearch(config);
	break;
    case TeachingWordConditionedTreeSearchType:
	recognizer = new Teaching::WordConditionedTreeSearch(config);
	break;
    default:
	Core::Application::us()->criticalError("unknown recognizer type: %d", type);
    }
    return recognizer;
}
