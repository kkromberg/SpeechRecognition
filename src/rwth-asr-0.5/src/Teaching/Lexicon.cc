#include "Lexicon.hh"

using namespace Teaching;

Lexicon::Lexicon(Bliss::LexiconRef lexicon, AcousticModelRef am) :
    lexicon_(lexicon),
    acousticModel_(am),
    pronunciations_(lexicon_->lemmaPronunciations().first)
{
    LemmaPronunciationIterator si = std::find(lexicon_->lemmaPronunciations().first, lexicon_->lemmaPronunciations().second,
					      lexicon_->specialLemma("silence")->pronunciations().first);
    verify(si != lexicon_->lemmaPronunciations().second);
    silence_ = std::distance(pronunciations_, si);
}

unsigned int Lexicon::nWords() const
{
    return lexicon_->nLemmaPronunciations();
}

unsigned int Lexicon::nPhonemes(Word w) const
{
    return pronunciation(w)->pronunciation()->length();
}

unsigned int Lexicon::nStates(Phoneme p) const
{
    return acousticModel_->hmmTopology(p)->nPhoneStates();
}

Phoneme Lexicon::getPhoneme(Word w, unsigned int position) const
{
    return (*pronunciation(w)->pronunciation())[position];
}

std::string Lexicon::symbol(Word w) const
{
    return pronunciation(w)->lemma()->name();
}

Word Lexicon::silence() const
{
    return silence_;
}

bool Lexicon::getMixtureSequence(Word w, MixtureSequence &mixtures) const
{
    mixtures.clear();
    const Am::Phonology &phonology = *acousticModel_->phonology();
    Core::Ref<const Am::AllophoneAlphabet> allophoneAlphabet = acousticModel_->allophoneAlphabet();
    Core::Ref<const Am::AllophoneStateAlphabet> allophoneStateAlphabet = acousticModel_->allophoneStateAlphabet();
    const Bliss::Pronunciation *pron = pronunciation(w)->pronunciation();
    for(unsigned short p = 0; p < pron->length(); ++p) {
	s16 boundary = 0;
	if(p == 0)
	    boundary |= Am::Allophone::isInitialPhone;
	if(p == pron->length() -1)
	    boundary |= Am::Allophone::isFinalPhone;

	const Am::Allophone *allo = allophoneAlphabet->allophone( Am::Allophone(phonology(*pron, p), boundary) );
	verify(allo);

	const Am::ClassicHmmTopology *hmmTopology = acousticModel_->hmmTopology( (*pron)[p] );
	verify(hmmTopology);

	for(int s = 0; s < hmmTopology->nPhoneStates(); ++s) {
	    Am::AllophoneState allophoneState = allophoneStateAlphabet->allophoneState(allo, s);
	    Mixture mixture = acousticModel_->emissionIndex(allophoneState);
	    for(int t = 0; t < hmmTopology->nSubStates(); ++t) {
		mixtures.push_back( mixture );
	    }
	}
    }
    return true;
}

const Mixture Lexicon::silenceMixture() const {
	return acousticModel_->emissionIndex(acousticModel_->silenceAllophoneStateIndex());
}

const Am::Allophone* Lexicon::allophone(Word word, Bliss::Phoneme::Id id) const {
	// boundary needs to be set for comparison of allophones
	int boundary = Am::Allophone::isWithinPhone;
	if (id == 0)
		boundary |= Am::Allophone::isInitialPhone;
	if (id == nPhonemes(word) - 1)
		boundary |= Am::Allophone::isFinalPhone;

	// obtain Allophone reference from AllophoneAlphabet (this is the only way)
	const Am::Phonology &phonology = *acousticModel_->phonology();
	const Bliss::Pronunciation &pron = *pronunciation(word)->pronunciation();

	return acousticModel_->allophoneAlphabet()->allophone(
			Am::Allophone(phonology(pron, id), boundary));
}

const MixtureSequence* Lexicon::mixtures(Word word, Bliss::Phoneme::Id id) const {
	MixtureSequence *mixtures = new MixtureSequence();

	const Bliss::Pronunciation &pron = *pronunciation(word)->pronunciation();
	const Am::ClassicHmmTopology *hmmTopology = acousticModel_
			->hmmTopology(pron[id]);

	// for all states of phoneme id
	for (int i = 0; i < hmmTopology->nPhoneStates(); i++) {
		Am::AllophoneState allophoneState = acousticModel_->allophoneStateAlphabet()
				->allophoneState(allophone(word, id), i);
		Mixture mixture = acousticModel_->emissionIndex(allophoneState);

		// add this mixture for every state repetition
		for (int j = 0; j < hmmTopology->nSubStates(); j++)
			mixtures->push_back(mixture);
	}

	return mixtures;
}
