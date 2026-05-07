#pragma once

#include "IScheduler.h"

// Sequential scheduler implementation.
// It preserves the parallel-style API while executing chunks on the calling thread.
class SequentialScheduler : public IScheduler
{
public:
    void parallelFor(
        size_t begin,
        size_t end,
        size_t grainSize,
        function<void(size_t, size_t)> task
    ) override;
};

