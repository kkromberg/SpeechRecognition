#ifndef _TEACHING_TYPES_HH
#define _TEACHING_TYPES_HH

#include <vector>
#include <limits>

namespace Teaching
{
    typedef unsigned int   Time;
    typedef unsigned short Mixture;
    typedef unsigned int   Word;
    typedef unsigned short Phoneme;
    typedef unsigned short  State;
    typedef unsigned int   Index;
    typedef std::vector<Word>    WordSequence;
    typedef std::vector<Mixture> MixtureSequence;
    typedef float Score;

    static const Word  invalidWord  = std::numeric_limits<Word>::max();
    static const Index invalidIndex = std::numeric_limits<Index>::max();
    static const Score maxScore     = std::numeric_limits<Score>::max();
}

#endif // _TEACHING_TYPES_HH
