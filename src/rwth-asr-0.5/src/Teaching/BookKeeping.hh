#ifndef _TEACHING_BOOK_KEEPING_HH
#define _TEACHING_BOOK_KEEPING_HH

#include "Types.hh"
#include "SearchInterface.hh"

namespace Teaching
{
    class BookKeeping
    {
    public:
	struct Entry
	{
	    Word  word;
	    Score score;
	    Index backpointer;
	    Time  time;
	    Time  timestamp;

	    Entry(Word w, Score s, Index b, Time t) :
		word(w), score(s), backpointer(b), time(t), timestamp(-1) {}

	    Entry() :
		word(invalidWord), score(maxScore), backpointer(0),
		time(0), timestamp(-1) {}

	    void set(Word w, Score s, Index b, Time t) {
		word = w; score = s; backpointer = b; time = t;
	    }

	    operator SearchInterface::TracebackItem() {
		return SearchInterface::TracebackItem(word, score, time);
	    }
	};

    public:
	BookKeeping()
	    : next_(0), lastTimestamp_(0) {}

	void clear();

	Index size() const {
	    return book_.size();
	}

	Entry &entry(Index backpointer) {
	    return book_[backpointer];
	}

	const Entry& entry(Index backpointer) const {
	    return book_[backpointer];
	}

	Index addEntry(Word word, Score score, Index backpointer, Time t);

	template<class HypothesisIterator>
	void tagActiveEntries(Time t, HypothesisIterator begin, HypothesisIterator end);

    private:
	Index findFreeStorage();
	void tagActiveEntries(Time t, Index backpointer);

	typedef std::vector<Entry> Book;
	Book  book_;
	Index next_;
	Time  lastTimestamp_;
	static const Index storageIncrement_, sentinelBackpointer_;
    };


    template<class HypothesisIterator>
    void BookKeeping::tagActiveEntries(Time t, HypothesisIterator hypBegin, HypothesisIterator hypEnd)
    {
	for (HypothesisIterator s = hypBegin; s != hypEnd; ++s)
	    tagActiveEntries(t, s->backpointer);
    }

}

#endif // _TEACHING_BOOK_KEEPING_HH
