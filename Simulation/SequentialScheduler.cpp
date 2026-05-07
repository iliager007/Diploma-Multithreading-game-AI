#include "SequentialScheduler.h"

#include <algorithm>

using std::min;

void SequentialScheduler::parallelFor(
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

    for (size_t chunkStart = begin; chunkStart < end; chunkStart += chunkSize)
    {
        const size_t chunkEnd = min(chunkStart + chunkSize, end);
        task(chunkStart, chunkEnd);
    }
}

