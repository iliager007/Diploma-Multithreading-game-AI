#include "ThreadPoolScheduler.h"

#include <algorithm>

using std::min;
using std::unique_lock;
using std::lock_guard;

ThreadPoolScheduler::ThreadPoolScheduler(size_t workerCount)
    : remainingTasks(0), stopping(false)
{
    const size_t actualWorkerCount = workerCount == 0 ? 1 : workerCount;
    workers.reserve(actualWorkerCount);

    for (size_t i = 0; i < actualWorkerCount; ++i)
    {
        workers.emplace_back(&ThreadPoolScheduler::workerLoop, this);
    }
}

ThreadPoolScheduler::~ThreadPoolScheduler()
{
    {
        lock_guard<mutex> lock(queueMutex);
        stopping = true;
    }

    taskAvailable.notify_all();

    for (thread& worker : workers)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

void ThreadPoolScheduler::parallelFor(
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

    {
        lock_guard<mutex> lock(queueMutex);

        // Each queued item represents a coarse chunk of agents.
        for (size_t chunkStart = begin; chunkStart < end; chunkStart += chunkSize)
        {
            const size_t chunkEnd = min(chunkStart + chunkSize, end);

            tasks.emplace_back([this, task, chunkStart, chunkEnd]() {
                task(chunkStart, chunkEnd);

                if (remainingTasks.fetch_sub(1) == 1)
                {
                    lock_guard<mutex> lock(completionMutex);
                    tasksFinished.notify_one();
                }
            });
        }
    }

    taskAvailable.notify_all();

    // The caller waits without busy spinning until workers finish all chunks.
    unique_lock<mutex> lock(completionMutex);
    tasksFinished.wait(lock, [this]() {
        return remainingTasks.load() == 0;
    });
}

size_t ThreadPoolScheduler::getWorkerCount() const
{
    return workers.size();
}

void ThreadPoolScheduler::workerLoop()
{
    while (true)
    {
        function<void()> task;

        {
            unique_lock<mutex> lock(queueMutex);

            // Workers sleep when no tasks are available and wake when a
            // parallelFor call pushes new chunks or the scheduler is stopping.
            taskAvailable.wait(lock, [this]() {
                return stopping || !tasks.empty();
            });

            if (stopping && tasks.empty())
            {
                return;
            }

            task = tasks.front();
            tasks.pop_front();
        }

        task();
    }
}
