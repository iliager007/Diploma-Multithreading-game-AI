#include "Simulation.h"
#include "SequentialScheduler.h"
#include "ThreadPoolScheduler.h"

#include <cstddef>
#include <iostream>
#include <thread>

using std::cout;
using std::size_t;
using std::thread;

void runBenchmarks(
    const char* schedulerName,
    size_t workerCount,
    IScheduler& scheduler,
    float heavyAgentRatio,
    int heavyWorkIterations,
    size_t grainSize
)
{
    constexpr float worldWidth = 1000.0f;
    constexpr float worldHeight = 1000.0f;
    constexpr int benchmarkFrames = 200;
    constexpr size_t agentCounts[] = {100, 1000, 5000, 10000, 50000};

    cout << "Scheduler: " << schedulerName << '\n';
    cout << "Workers: " << workerCount << "\n\n";

    for (const size_t agentCount : agentCounts)
    {
        Simulation simulation(
            worldWidth,
            worldHeight,
            &scheduler,
            grainSize,
            heavyAgentRatio,
            heavyWorkIterations
        );
        simulation.initialize(agentCount);

        const BenchmarkResult result = simulation.runBenchmark(benchmarkFrames);

        cout << "Agents: " << simulation.getWorld().getAgents().size() << '\n';
        cout << "Heavy ratio: " << heavyAgentRatio << '\n';
        cout << "Heavy iterations: " << heavyWorkIterations << '\n';
        cout << "Grain size: " << grainSize << '\n';
        cout << "Avg frame time: " << result.averageFrameTimeMs << " ms\n";
        cout << "Min frame time: " << result.minFrameTimeMs << " ms\n";
        cout << "Max frame time: " << result.maxFrameTimeMs << " ms\n\n";
    }
}

int main()
{
    const unsigned int hardwareWorkers = thread::hardware_concurrency();
    const size_t workerCount = hardwareWorkers == 0 ? 2 : hardwareWorkers;
    constexpr float heavyAgentRatio = 0.10f;
    constexpr int heavyWorkIterations = 100;
    constexpr size_t grainSize = 1;

    SequentialScheduler sequentialScheduler;
    ThreadPoolScheduler threadPoolScheduler(workerCount);

    runBenchmarks("SequentialScheduler", 1, sequentialScheduler, heavyAgentRatio, heavyWorkIterations, grainSize);
    runBenchmarks(
        "ThreadPoolScheduler",
        threadPoolScheduler.getWorkerCount(),
        threadPoolScheduler,
        heavyAgentRatio,
        heavyWorkIterations,
        grainSize
    );

    return 0;
}
