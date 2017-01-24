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
#include "Thread.hh"

using namespace Core;

/**
 * Start the thread
 */
bool Thread::start()
{
    int retval = pthread_create(&thread_, 0, Thread::startRun, static_cast<void*>(this));
    if (retval != 0)
	return false;
    running_ = true;
    return true;
}

/**
 * Wait for thread to exit
 */
void Thread::wait()
{
    if(this->running_) {
	pthread_join(thread_, 0);
    }
}

/**
 * Exit from Thread
 */
void Thread::exitThread()
{
    if(this->running_)
	pthread_exit(0);
    running_ = false;
}

inline void Thread::enableCancel()
{
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
}

inline void Thread::disableCancel()
{
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
}

inline void Thread::setCancelDeferred()
{
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
}

inline void Thread::setCancelAsync()
{
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, 0);
}


void Thread::startCleanup(void *obj)
{
    Thread *curObj = static_cast<Thread *>(obj);
    curObj->cleanup();
}

void* Thread::startRun(void *obj)
{
    Thread *curObj = static_cast<Thread *>(obj);
    pthread_cleanup_push(Thread::startCleanup, obj);
    curObj->run();
    pthread_cleanup_pop(0);
    return 0;
}
