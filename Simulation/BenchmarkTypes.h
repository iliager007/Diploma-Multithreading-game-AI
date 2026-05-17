#pragma once

#include <cstddef>
#include <string>

using std::size_t;
using std::string;

// Full configuration for one benchmark row.
struct BenchmarkConfig
{
    string schedulerName;

    size_t agentCount;
    size_t workerCount;
    size_t grainSize;

    float heavyAgentRatio;
    int heavyWorkIterations;

    int benchmarkFrames;

    string suiteName = "default";
};

// Timing plus scalability metrics for one benchmark run.
struct BenchmarkResult
{
    double averageFrameTimeMs;
    double minFrameTimeMs;
    double maxFrameTimeMs;

    double speedup;
    double efficiency;

    double heavyLoad = 0.0;
    double throughputAgentsPerSec = 0.0;
};
