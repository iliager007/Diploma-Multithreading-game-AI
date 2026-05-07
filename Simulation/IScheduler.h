#pragma once

#include <cstddef>
#include <functional>

using std::function;
using std::size_t;

// Abstract loop scheduler.
class IScheduler
{
public:
    virtual ~IScheduler() = default;

    // Processes [begin, end) in subranges
    virtual void parallelFor(
        size_t begin,
        size_t end,
        size_t grainSize,
        function<void(size_t, size_t)> task
    ) = 0;
};

