#pragma once

#include "IScheduler.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

using std::atomic;
using std::condition_variable;
using std::deque;
using std::mt19937;
using std::mutex;
using std::size_t;
using std::thread;
using std::unique_ptr;
using std::vector;

// Work-stealing scheduler for uneven agent workloads.
class WorkStealingScheduler : public IScheduler
{
public:
    explicit WorkStealingScheduler(size_t workerCount);
    ~WorkStealingScheduler() override;

    void parallelFor(
        size_t begin,
        size_t end,
        size_t grainSize,
        function<void(size_t, size_t)> task
    ) override;

    size_t getWorkerCount() const;

private:
    using Task = function<void()>;

    struct WorkerQueue
    {
        deque<Task> tasks;
        mutex mutex;
    };

    void workerLoop(size_t workerIndex);
    bool tryPopLocal(size_t workerIndex, Task& task);
    bool trySteal(size_t workerIndex, mt19937& rng, Task& task);
    void finishTask();

    vector<unique_ptr<WorkerQueue>> workerQueues;
    vector<thread> workers;

    mutex sleepMutex;
    mutex completionMutex;
    condition_variable workAvailable;
    condition_variable tasksFinished;

    atomic<size_t> queuedTasks;
    atomic<size_t> remainingTasks;
    atomic<bool> submittingTasks;
    atomic<bool> stopping;
};

