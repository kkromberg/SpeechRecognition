#include "BookKeeping.hh"

using namespace Teaching;

const Index BookKeeping::storageIncrement_    = 512;
const Index BookKeeping::sentinelBackpointer_ = 0;

void BookKeeping::clear()
{
    book_.clear();
    book_.resize(storageIncrement_);
    next_ = sentinelBackpointer_;
    lastTimestamp_ = 0;
}

Index BookKeeping::addEntry(Word word, Score score, Index backpointer, Time t)
{
    Index b = findFreeStorage();
    book_[b].set(word, score, backpointer, t);
    return b;
}

Index BookKeeping::findFreeStorage()
{
    Index maxIndex = book_.size() - 1, cnt = 0;
    next_ = next_ % maxIndex + 1;
    while(book_[next_].timestamp == lastTimestamp_ && cnt <= maxIndex) {
	next_ = next_ % maxIndex + 1;
	++cnt;
    }
    if (book_[next_].timestamp == lastTimestamp_) {
	// no free space found
	next_ = maxIndex + 1;
	book_.resize(book_.size() + storageIncrement_);
    }
    book_[next_].timestamp = lastTimestamp_;
    return next_;
}

void BookKeeping::tagActiveEntries(Time t, Index backpointer)
{
    lastTimestamp_ = t;
    while(book_[backpointer].timestamp != lastTimestamp_ && book_[backpointer].backpointer != sentinelBackpointer_) {
	book_[backpointer].timestamp = lastTimestamp_;
	backpointer = book_[backpointer].backpointer;
    }
}

