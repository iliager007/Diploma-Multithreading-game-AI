#pragma once

#include "IScheduler.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

using std::atomic;
using std::condition_variable;
using std::deque;
using std::mutex;
using std::size_t;
using std::thread;
using std::vector;

// Reusable thread pool scheduler.
class ThreadPoolScheduler : public IScheduler
{
public:
    explicit ThreadPoolScheduler(size_t workerCount);
    ~ThreadPoolScheduler() override;

    void parallelFor(
        size_t begin,
        size_t end,
        size_t grainSize,
        function<void(size_t, size_t)> task
    ) override;

    size_t getWorkerCount() const;

private:
    void workerLoop();

    vector<thread> workers;
    deque<function<void()>> tasks;

    mutex queueMutex;
    mutex completionMutex;
    condition_variable taskAvailable;
    condition_variable tasksFinished;

    atomic<size_t> remainingTasks;
    bool stopping;
};
