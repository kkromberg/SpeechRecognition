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
// $Id: Phoneme.cc 6223 2006-11-14 17:01:38Z rybach $
#include <Core/Utility.hh>
#include "Phoneme.hh"
#include "Fsa.hh"

using namespace Bliss;


const Phoneme::Id Phoneme::term;

Phoneme::Phoneme() :
    Token(),
    isContextDependent_(true)
{}

struct PhonemeInventory::Internal {
    SymbolSet symbols_;
    Core::WeakRef<const PhonemeAlphabet> phonemeAlphabet_;
};

PhonemeInventory::PhonemeInventory() :
    internal_(0)
{
    internal_ = new Internal;
    Phoneme *term = new Phoneme();
    term->setSymbol(internal_->symbols_["__term__"]);
    phonemes_.insert(term);
}

#ifdef DEPRECATED
PhonemeInventory::PhonemeInventory(const PhonemeInventory &phonemeInventory) {
    PhonemeIterator pi, pi_end;
    for (Core::tie(pi, pi_end) = phonemeInventory.phonemes(); pi != pi_end; ++pi) {
	Phoneme *phoneme = newPhoneme();
	phoneme->isContextDependent_ = (*pi)->isContextDependent();
	assignSymbol(phoneme, (*pi)->symbol());
    }
}
#endif

PhonemeInventory::~PhonemeInventory() {
    delete internal_;
}

Phoneme *PhonemeInventory::newPhoneme() {
    Phoneme *pho = new Phoneme;
    phonemes_.insert(pho);
    return pho ;
}

void PhonemeInventory::assignSymbol(Phoneme *pho, const std::string &sym) {
    require(pho);
    require(!phoneme(sym));

    Symbol symbol = internal_->symbols_[sym];
    if (!pho->symbol())	pho->setSymbol(symbol);
    phonemes_.link(symbol, pho);
}

void PhonemeInventory::writeXml(Core::XmlWriter &os) const {
    os << Core::XmlOpen("phoneme-inventory");
    PhonemeIterator pi, pi_end;
    for (Core::tie(pi, pi_end) = phonemes(); pi != pi_end; ++pi) {
	os << Core::XmlOpen("phoneme")
	   << Core::XmlFull("symbol", (*pi)->symbol());
	if ((*pi)->isContextDependent())
	    os << Core::XmlFull("variation", "context");
	else
	    os << Core::XmlFull("variation", "none");
	os << Core::XmlClose("phoneme");
    }
    os << Core::XmlClose("phoneme-inventory");
}

/**
 * \todo PhonemeInventory::writeBinary not implemented
 */
void PhonemeInventory::writeBinary(Core::BinaryOutputStream &os) const {
    defect();
}

Core::Ref<const PhonemeAlphabet> PhonemeInventory::phonemeAlphabet() const {
    Core::Ref<const PhonemeAlphabet> result;
    if (internal_->phonemeAlphabet_)
	result = internal_->phonemeAlphabet_;
    else
	internal_->phonemeAlphabet_ = result = Core::ref(new PhonemeAlphabet(Core::ref(this)));
    return result;
}
