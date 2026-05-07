#pragma once

#include "IScheduler.h"
#include "World.h"

#include <cstddef>
#include <random>

using std::mt19937;
using std::size_t;

// Aggregated timing data for a benchmark run.
struct BenchmarkResult
{
    double averageFrameTimeMs;
    double minFrameTimeMs;
    double maxFrameTimeMs;
};

// Coordinates the world and owns the simulation loop.
class Simulation
{
public:
    Simulation(
        float worldWidth,
        float worldHeight,
        IScheduler* scheduler,
        size_t grainSize,
        float heavyAgentRatio,
        int heavyWorkIterations
    );

    // Initializes world state using a fixed-seed random generator.
    void initialize(size_t agentCount);

    // Runs update steps without printing benchmark output.
    void run(size_t steps);

    // Measures one full update tick in milliseconds.
    double update(float deltaTime, float speed);

    // Runs several timed frames and returns stable aggregate timing statistics.
    BenchmarkResult runBenchmark(int frameCount);

    const World& getWorld() const;

private:
    void updateAgents(float deltaTime, float speed);

    World world;
    IScheduler* scheduler;
    size_t grainSize;
    float heavyAgentRatio;
    int heavyWorkIterations;
    mt19937 rng;
};
