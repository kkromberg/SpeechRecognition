#ifndef _TEACHING_LEXICON_HH
#define _TEACHING_LEXICON_HH

#include "Types.hh"
#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>

namespace Teaching
{
    /**
     * provides access to the pronunciation dictionary and the HMM models.
     * simple interface to Bliss::Lexicon and the Am::AcousticModel.
     */
    class Lexicon
    {
	typedef Core::Ref<const Am::AcousticModel>         AcousticModelRef;
	typedef Bliss::Lexicon::LemmaPronunciationIterator LemmaPronunciationIterator;
	typedef Bliss::LemmaPronunciation                  LemmaPronunciation;

    public:
	Lexicon(Bliss::LexiconRef lexicon, AcousticModelRef am);

	/**
	 * number of words.
	 * in fact, number of distinct lemma pronunciations, but to keep things
	 * simple each lemma pronunciation is considered as a separate word
	 */
	unsigned int nWords() const;

	/**
	 * number of phonemes of word w
	 */
	unsigned int nPhonemes(Word w) const;

	/**
	 * number of states of phoneme p
	 */
	unsigned int nStates(Phoneme p) const;

	/**
	 * string representation of word w
	 */
	std::string symbol(Word w) const;

	/**
	 * i-th phoneme in word w
	 */
	Phoneme getPhoneme(Word w, unsigned int position) const;

	/**
	 * silence "word"
	 */
	Word silence() const;

	/**
	 * sequence of mixtures for states of word w
	 * @param w word
	 * @parm  result mixture sequence for word w
	 * @return ok
	 */
	bool getMixtureSequence(Word w, MixtureSequence &result) const;

	/**
	 * interface to the Bliss module.
	 * convert word index to LemmaPronunciation pointer
	 */
	const Bliss::LemmaPronunciation* pronunciation(Word w) const {
	    return *(pronunciations_ + w);
	}

    /**
     * Return mixture corresponding to silence phoneme (modelled by
     * a single state).
     */
	const Mixture silenceMixture() const;

	/**
	 * Return allophone given word and phoneme id.
	 */
	const Am::Allophone* allophone(Word word, Bliss::Phoneme::Id id) const;

	/**
	 * Return mixture sequence for given word and phoneme id.
	 */
	const MixtureSequence* mixtures(Word word, Bliss::Phoneme::Id id) const;

	/**
	 * Return string representation of given allophone.
	 */
	inline std::string format(const Am::Allophone &allophone) const {
	    return allophone.format(lexicon_->phonemeInventory());
	}

    protected:

    protected:
	Bliss::LexiconRef lexicon_;
	AcousticModelRef acousticModel_;
	LemmaPronunciationIterator pronunciations_;
	Word silence_;

    };

}

#endif // _TEACHING_LEXICON_HH
