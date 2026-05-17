#include "Simulation.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>

using std::clamp;
using std::chrono::duration;
using std::chrono::high_resolution_clock;
using std::milli;
using std::numeric_limits;
using std::invalid_argument;
using std::size_t;
using std::vector;

Simulation::Simulation(
    float worldWidth,
    float worldHeight,
    IScheduler* scheduler,
    size_t grainSize,
    float heavyAgentRatio,
    int heavyWorkIterations
)
    : world(worldWidth, worldHeight),
      scheduler(scheduler),
      grainSize(grainSize == 0 ? 1 : grainSize),
      heavyAgentRatio(clamp(heavyAgentRatio, 0.0f, 1.0f)),
      heavyWorkIterations(heavyWorkIterations < 0 ? 0 : heavyWorkIterations),
      rng(42)
{
    if (scheduler == nullptr)
    {
        throw invalid_argument("Simulation requires a scheduler.");
    }
}

void Simulation::initialize(size_t agentCount)
{
    world.initializeAgents(agentCount, rng, heavyAgentRatio);
}

void Simulation::run(size_t steps)
{
    constexpr float deltaTime = 0.016f;
    constexpr float speed = 5.0f;

    for (size_t step = 0; step < steps; ++step)
    {
        updateAgents(deltaTime, speed);
    }
}

double Simulation::update(float deltaTime, float speed)
{
    // Timing covers one complete simulation tick: all agents updated once.
    const auto startTime = high_resolution_clock::now();
    updateAgents(deltaTime, speed);
    const auto endTime = high_resolution_clock::now();

    const duration<double, milli> elapsedMilliseconds = endTime - startTime;
    return elapsedMilliseconds.count();
}

void Simulation::updateAgents(float deltaTime, float speed)
{
    vector<Agent>& agents = world.getAgents();
    const float worldWidth = world.getWidth();
    const float worldHeight = world.getHeight();

    scheduler->parallelFor(0, agents.size(), grainSize, [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i)
        {
            agents[i].update(deltaTime, speed, worldWidth, worldHeight, heavyWorkIterations);
        }
    });
}

BenchmarkResult Simulation::runBenchmark(int frameCount)
{
    constexpr float deltaTime = 0.016f;
    constexpr float speed = 5.0f;
    constexpr int minFrames = 100;
    constexpr int maxFrames = 1000;

    const int measuredFrames = clamp(frameCount, minFrames, maxFrames);

    vector<double> frameTimes;
    frameTimes.reserve(static_cast<size_t>(measuredFrames));

    for (int frame = 0; frame < measuredFrames; ++frame)
    {
        frameTimes.push_back(update(deltaTime, speed));
    }

    double totalMilliseconds = 0.0;
    double minMilliseconds = numeric_limits<double>::max();
    double maxMilliseconds = numeric_limits<double>::lowest();

    for (const double frameTime : frameTimes)
    {
        totalMilliseconds += frameTime;
        minMilliseconds = frameTime < minMilliseconds ? frameTime : minMilliseconds;
        maxMilliseconds = frameTime > maxMilliseconds ? frameTime : maxMilliseconds;
    }

    return {
        totalMilliseconds / static_cast<double>(measuredFrames),
        minMilliseconds,
        maxMilliseconds,
        0.0,
        0.0
    };
}

const World& Simulation::getWorld() const
{
    return world;
}
