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
#ifndef _CORE_THREAD_HH
#define _CORE_THREAD_HH

#include <pthread.h>
#include <sys/time.h>
#include <ctime>

#ifndef PTHREAD_MUTEX_RECURSIVE_NP
// PTHREAD_MUTEX_RECURSIVE_NP is not available on some plattforms
#define PTHREAD_MUTEX_RECURSIVE_NP PTHREAD_MUTEX_RECURSIVE
#endif

namespace Core {

    /**
     * Abstract Thread.
     *
     * Thread implementation based on pthreads
     */
    class Thread
    {
    public:
	Thread()
	    : running_(false)
	{}

	virtual ~Thread() {}

	/**
	 * Start the thread
	 */
	bool start();

	/**
	 * Wait for thread to exit
	 */
	void wait();


    protected:
	/**
	 * Allow that the thread is cancelled by another thread
	 */
	void enableCancel();
	/**
	 * Forbit other threads to cancel this thread
	 */
	void disableCancel();
	/**
	 * The thread can only be chancelled on certain points
	 */
	void setCancelDeferred();
	/**
	 * The thread can be chanceled at any time
	 */
	void setCancelAsync();

	/**
	 * Override this in a threaded class
	 */
	virtual void run() {}
	/**
	 * Cleanup, when thread is ended
	 */
	virtual void cleanup() {}

	/**
	 * Terminates the thread
	 */
	void exitThread();

    private:
	static void* startRun(void *);
	static void startCleanup(void *);
	bool running_;
	pthread_t thread_;
    };


    /**
     * A mutex.
     *
     * A mutex is a MUTual EXclusion device, and is useful for protecting
     * shared data structures from concurrent modifications, and implementing
     * critical sections and monitors
     */
    class Mutex
    {
    private:
	// disable copying
	Mutex(const Mutex&);
	Mutex& operator=(const Mutex&);
    public:
	Mutex() {
	    pthread_mutexattr_t attr;
	    pthread_mutexattr_init(&attr);
	    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE_NP);
	    pthread_mutex_init(&mutex_, &attr);
	    pthread_mutexattr_destroy(&attr);
	}
	Mutex(const pthread_mutex_t &m)
	    : mutex_(m) {}

	~Mutex() {
	    pthread_mutex_destroy(&mutex_);
	}

	bool lock() {
	    return (pthread_mutex_lock(&mutex_) == 0);
	}

	bool unlock() {
	    return (pthread_mutex_unlock(&mutex_) == 0);
	}

	bool release() {
	    return unlock();
	}

	bool trylock() {
	    return (pthread_mutex_trylock(&mutex_) == 0);
	}

    private:
	pthread_mutex_t mutex_;
	friend class Condition;
    };


    /**
     * A read-write lock
     */
    class ReadWriteLock
    {
    private:
	ReadWriteLock(const ReadWriteLock&);
	ReadWriteLock& operator=(const ReadWriteLock&);
    public:
	ReadWriteLock() {
	    pthread_rwlock_init(&lock_, 0);
	}

	~ReadWriteLock() {
	    pthread_rwlock_destroy(&lock_);
	}

	/**
	 * Acquire a read lock
	 */
	bool readLock() {
	    return (pthread_rwlock_rdlock(&lock_) == 0);
	}
	/**
	 * try to acquire a read lock
	 */
	bool tryReadLock() {
	    return (pthread_rwlock_tryrdlock(&lock_) == 0);
	}
	/**
	 * Acquire a write lock
	 */
	bool writeLock() {
	    return (pthread_rwlock_wrlock(&lock_) == 0);
	}
	/**
	 * Try to acquire a write lock
	 */
	bool tryWriteLock() {
	    return (pthread_rwlock_trywrlock(&lock_) == 0);
	}
	/**
	 * Unlock
	 */
	bool unlock() {
	    return (pthread_rwlock_unlock(&lock_) == 0);
	}
    private:
	pthread_rwlock_t lock_;
    };


    /**
     * A condition variable.
     *
     * A condition variable is a synchronization device that allows
     * threads to suspend execution and relinquish the processors
     * until some predicate on shared data is satisfied.
     */
    class Condition
    {
    private:
	Condition(const Condition&);
	Condition& operator=(const Condition&);
    public:
	Condition() {
	    pthread_cond_init(&cond_, 0);
	}

	~Condition() {
	    pthread_cond_destroy(&cond_);
	}

	/**
	 * Restarts one of the threads that are waiting on the condition
	 * variable.
	 *
	 * @param broadcastSignal Restarts all waiting threads
	 */
	bool signal(bool broadcastSignal = false) {
	    if (broadcastSignal)
		return broadcast();
	    return(pthread_cond_signal(&cond_) == 0);
	}

	/**
	 * Restart all waiting threads.
	 * @see signal
	 */
	bool broadcast() {
	    return (pthread_cond_broadcast(&cond_) == 0);
	}

	/**
	 * Wait for the condition to be signaled.
	 *
	 * The thread execution is suspended and does not consume any CPU time
	 * until the condition variable is signaled.
	 */
	bool wait() {
	    mutex_.lock();
	    bool r = (pthread_cond_wait(&cond_, &mutex_.mutex_) == 0);
	    mutex_.unlock();
	    return r;
	}

	/**
	 * Wait for the condition to be signaled with a bounded wait time.
	 * @see wait
	 */
	bool timedWait(unsigned long microsonds) {
	    mutex_.lock();
	    timespec timeout = getTimeout(microsonds);
	    bool r = (pthread_cond_timedwait(&cond_, &mutex_.mutex_, &timeout) == 0);
	    mutex_.unlock();
	    return r;
	}

    private:
	timespec getTimeout(unsigned long microseconds) {
	    timeval now;
	    gettimeofday(&now, 0);
	    timespec timeout;
	    timeout.tv_sec = now.tv_sec + microseconds / 1000000;
	    timeout.tv_nsec = now.tv_usec * 1000 + microseconds % 1000000;
	    return timeout;
	}

	pthread_cond_t cond_;

	/**
	 * A condition variable must always be associated with a mutex,
	 * to avoid the race condition where a thread prepares to wait
	 * on a condition variable and another thread signals the condition
	 * just before the first thread actually waits on it.
	 */
	Mutex mutex_;
    };

}

#endif // _CORE_THREAD_HH
