#include "WorkStealingScheduler.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>

using std::lock_guard;
using std::make_unique;
using std::min;
using std::move;
using std::uint32_t;
using std::uniform_int_distribution;
using std::unique_lock;

WorkStealingScheduler::WorkStealingScheduler(size_t workerCount)
    : queuedTasks(0),
      remainingTasks(0),
      submittingTasks(false),
      stopping(false)
{
    const size_t actualWorkerCount = workerCount == 0 ? 1 : workerCount;
    workerQueues.reserve(actualWorkerCount);
    workers.reserve(actualWorkerCount);

    for (size_t i = 0; i < actualWorkerCount; ++i)
    {
        workerQueues.emplace_back(make_unique<WorkerQueue>());
    }

    for (size_t i = 0; i < actualWorkerCount; ++i)
    {
        workers.emplace_back(&WorkStealingScheduler::workerLoop, this, i);
    }
}

WorkStealingScheduler::~WorkStealingScheduler()
{
    stopping.store(true);
    workAvailable.notify_all();

    for (thread& worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

void WorkStealingScheduler::parallelFor(
    size_t begin,
    size_t end,
    size_t grainSize,
    function<void(size_t, size_t)> task
)
{
    if (begin >= end)
    {
        return;
    }

    const size_t chunkSize = grainSize == 0 ? 1 : grainSize;
    const size_t itemCount = end - begin;
    const size_t chunkCount = (itemCount + chunkSize - 1) / chunkSize;

    remainingTasks.store(chunkCount);
    queuedTasks.store(0);
    submittingTasks.store(true);

    size_t chunkIndex = 0;

    for (size_t chunkStart = begin; chunkStart < end; chunkStart += chunkSize)
    {
        const size_t chunkEnd = min(chunkStart + chunkSize, end);
        const size_t queueIndex = chunkIndex % workerQueues.size();

        Task queuedTask = [this, task, chunkStart, chunkEnd]() {
            task(chunkStart, chunkEnd);
            finishTask();
        };

        {
            lock_guard<mutex> lock(workerQueues[queueIndex]->mutex);
            workerQueues[queueIndex]->tasks.emplace_back(move(queuedTask));
            queuedTasks.fetch_add(1);
        }

        ++chunkIndex;
    }

    submittingTasks.store(false);
    workAvailable.notify_all();

    unique_lock<mutex> lock(completionMutex);
    tasksFinished.wait(lock, [this]() {
        return remainingTasks.load() == 0;
    });
}

size_t WorkStealingScheduler::getWorkerCount() const
{
    return workers.size();
}

void WorkStealingScheduler::workerLoop(size_t workerIndex)
{
    mt19937 rng(static_cast<uint32_t>(0x9E3779B9u + workerIndex));

    while (true)
    {
        Task task;

        if (tryPopLocal(workerIndex, task) || trySteal(workerIndex, rng, task))
        {
            task();
            continue;
        }

        unique_lock<mutex> lock(sleepMutex);

        workAvailable.wait(lock, [this]() {
            return stopping.load() || (!submittingTasks.load() && queuedTasks.load() > 0);
        });

        if (stopping.load() && queuedTasks.load() == 0)
        {
            return;
        }
    }
}

bool WorkStealingScheduler::tryPopLocal(size_t workerIndex, Task& task)
{
    WorkerQueue& queue = *workerQueues[workerIndex];
    lock_guard<mutex> lock(queue.mutex);

    if (queue.tasks.empty())
    {
        return false;
    }

    // Owner workers use the back of their own deque for cache-friendly local work.
    task = move(queue.tasks.back());
    queue.tasks.pop_back();
    queuedTasks.fetch_sub(1);

    return true;
}

bool WorkStealingScheduler::trySteal(size_t workerIndex, mt19937& rng, Task& task)
{
    if (workerQueues.size() <= 1)
    {
        return false;
    }

    uniform_int_distribution<size_t> victimDistribution(0, workerQueues.size() - 1);

    // Stealing is attempted only after local work is exhausted
    for (size_t attempt = 0; attempt < workerQueues.size() * 2; ++attempt)
    {
        const size_t victimIndex = victimDistribution(rng);

        if (victimIndex == workerIndex)
        {
            continue;
        }

        WorkerQueue& victimQueue = *workerQueues[victimIndex];
        lock_guard<mutex> lock(victimQueue.mutex);

        if (!victimQueue.tasks.empty())
        {
            // Thieves take from the front, opposite of the owner
            task = move(victimQueue.tasks.front());
            victimQueue.tasks.pop_front();
            queuedTasks.fetch_sub(1);

            return true;
        }
    }

    return false;
}

void WorkStealingScheduler::finishTask()
{
    if (remainingTasks.fetch_sub(1) == 1)
    {
        lock_guard<mutex> lock(completionMutex);
        tasksFinished.notify_one();
    }
}
