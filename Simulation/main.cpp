#include "BenchmarkCSVExporter.h"
#include "BenchmarkTypes.h"
#include "SequentialScheduler.h"
#include "Simulation.h"
#include "ThreadPoolScheduler.h"
#include "WorkStealingScheduler.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using std::cout;
using std::make_unique;
using std::pair;
using std::size_t;
using std::string;
using std::unique_ptr;
using std::vector;

void appendSequentialConfig(
    vector<BenchmarkConfig>& configs,
    const string& suiteName,
    size_t agentCount,
    size_t grainSize,
    float heavyAgentRatio,
    int heavyWorkIterations,
    int benchmarkFrames
)
{
    configs.push_back({
        "Sequential",
        agentCount,
        1,
        grainSize,
        heavyAgentRatio,
        heavyWorkIterations,
        benchmarkFrames,
        suiteName
    });
}

void appendParallelConfigs(
    vector<BenchmarkConfig>& configs,
    const string& suiteName,
    const string& schedulerName,
    const vector<size_t>& workerCounts,
    size_t agentCount,
    size_t grainSize,
    float heavyAgentRatio,
    int heavyWorkIterations,
    int benchmarkFrames
)
{
    for (const size_t workerCount : workerCounts)
    {
        configs.push_back({
            schedulerName,
            agentCount,
            workerCount,
            grainSize,
            heavyAgentRatio,
            heavyWorkIterations,
            benchmarkFrames,
            suiteName
        });
    }
}

vector<BenchmarkConfig> createDefaultSuite()
{
    const vector<size_t> agentCounts = {1000, 5000, 10000, 20000, 50000};
    const vector<size_t> grainSizes = {16, 64, 256, 1024, 2048};
    const vector<float> heavyRatios = {0.0f, 0.1f, 0.25f};
    const vector<int> heavyIterations = {1000, 5000};
    const vector<size_t> workerCounts = {1, 2, 4, 8};
    constexpr int benchmarkFrames = 100;

    vector<BenchmarkConfig> configs;
    configs.reserve(agentCounts.size() * grainSizes.size() * heavyRatios.size() * heavyIterations.size() * 9);

    // Default suite preserves the existing benchmark workflow and baseline rule.
    for (const size_t agentCount : agentCounts)
    {
        for (const size_t grainSize : grainSizes)
        {
            for (const float heavyRatio : heavyRatios)
            {
                for (const int iterations : heavyIterations)
                {
                    appendSequentialConfig(configs, "default", agentCount, grainSize, heavyRatio, iterations, benchmarkFrames);
                    appendParallelConfigs(configs, "default", "ThreadPool", workerCounts, agentCount, grainSize, heavyRatio, iterations, benchmarkFrames);
                    appendParallelConfigs(configs, "default", "WorkStealing", workerCounts, agentCount, grainSize, heavyRatio, iterations, benchmarkFrames);
                }
            }
        }
    }

    return configs;
}

vector<BenchmarkConfig> createLightOverheadSuite()
{
    const vector<size_t> agentCounts = {10000, 50000, 100000, 250000, 500000, 1000000};
    const vector<size_t> grainSizes = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536};
    const vector<size_t> workerCounts = {1, 2, 4, 8};
    constexpr float heavyRatio = 0.0f;
    constexpr int heavyIterations = 0;
    constexpr int benchmarkFrames = 100;

    vector<BenchmarkConfig> configs;
    configs.reserve(agentCounts.size() * grainSizes.size() * 9);

    // This suite isolates scheduler overhead when each agent update is cheap.
    for (const size_t agentCount : agentCounts)
    {
        for (const size_t grainSize : grainSizes)
        {
            appendSequentialConfig(configs, "light_overhead", agentCount, grainSize, heavyRatio, heavyIterations, benchmarkFrames);
            appendParallelConfigs(configs, "light_overhead", "ThreadPool", workerCounts, agentCount, grainSize, heavyRatio, heavyIterations, benchmarkFrames);
            appendParallelConfigs(configs, "light_overhead", "WorkStealing", workerCounts, agentCount, grainSize, heavyRatio, heavyIterations, benchmarkFrames);
        }
    }

    return configs;
}

vector<BenchmarkConfig> createHeavyParallelSuite()
{
    const vector<size_t> agentCounts = {10000, 50000, 100000, 200000};
    const vector<size_t> grainSizes = {64, 256, 1024, 4096};
    const vector<float> heavyRatios = {0.01f, 0.10f, 0.25f};
    const vector<int> heavyIterations = {5000, 10000};
    const vector<size_t> workerCounts = {4, 8};
    constexpr int benchmarkFrames = 50;

    vector<BenchmarkConfig> configs;
    configs.reserve(agentCounts.size() * grainSizes.size() * heavyRatios.size() * heavyIterations.size() * 8);

    // Heavy parallel omits Sequential by design. Speedup is measured against
    // the same scheduler with one worker to compare scaling under imbalance.
    for (const size_t agentCount : agentCounts)
    {
        for (const size_t grainSize : grainSizes)
        {
            for (const float heavyRatio : heavyRatios)
            {
                for (const int iterations : heavyIterations)
                {
                    appendParallelConfigs(configs, "heavy_parallel", "ThreadPool", workerCounts, agentCount, grainSize, heavyRatio, iterations, benchmarkFrames);
                    appendParallelConfigs(configs, "heavy_parallel", "WorkStealing", workerCounts, agentCount, grainSize, heavyRatio, iterations, benchmarkFrames);
                }
            }
        }
    }

    return configs;
}

bool shouldSkipBenchmark(const BenchmarkConfig& config)
{
    return (config.agentCount >= 200000 && config.heavyAgentRatio >= 0.25f && config.heavyWorkIterations >= 25000)
        || (config.agentCount >= 100000 && config.heavyAgentRatio >= 0.50f && config.heavyWorkIterations >= 25000);
}

unique_ptr<IScheduler> createScheduler(const string& schedulerName, size_t workerCount)
{
    if (schedulerName == "Sequential")
    {
        return make_unique<SequentialScheduler>();
    }

    if (schedulerName == "ThreadPool")
    {
        return make_unique<ThreadPoolScheduler>(workerCount);
    }

    return make_unique<WorkStealingScheduler>(workerCount);
}

void logProgress(const BenchmarkConfig& config)
{
    cout << "[Suite=" << config.suiteName << "]\n"
         << "Scheduler=" << config.schedulerName << '\n'
         << "Threads=" << config.workerCount << '\n'
         << "Agents=" << config.agentCount << '\n'
         << "Grain=" << config.grainSize << '\n'
         << "HeavyRatio=" << config.heavyAgentRatio << '\n'
         << "HeavyIterations=" << config.heavyWorkIterations << "\n\n";
}

BenchmarkResult runBenchmark(const BenchmarkConfig& config)
{
    constexpr float worldWidth = 1000.0f;
    constexpr float worldHeight = 1000.0f;

    logProgress(config);

    unique_ptr<IScheduler> scheduler = createScheduler(config.schedulerName, config.workerCount);

    Simulation simulation(
        worldWidth,
        worldHeight,
        scheduler.get(),
        config.grainSize,
        config.heavyAgentRatio,
        config.heavyWorkIterations
    );
    simulation.initialize(config.agentCount);

    return simulation.runBenchmark(config.benchmarkFrames);
}

void applyDerivedMetrics(BenchmarkResult& result, const BenchmarkConfig& config, double baselineMs)
{
    result.speedup = result.averageFrameTimeMs > 0.0
        ? baselineMs / result.averageFrameTimeMs
        : 0.0;

    result.efficiency = config.workerCount <= 1
        ? 1.0
        : result.speedup / static_cast<double>(config.workerCount);

    result.heavyLoad = static_cast<double>(config.heavyAgentRatio) * static_cast<double>(config.heavyWorkIterations);
    result.throughputAgentsPerSec = result.averageFrameTimeMs > 0.0
        ? static_cast<double>(config.agentCount) / (result.averageFrameTimeMs / 1000.0)
        : 0.0;
}

bool sameDefaultScenario(const BenchmarkConfig& left, const BenchmarkConfig& right)
{
    return left.agentCount == right.agentCount
        && left.grainSize == right.grainSize
        && left.heavyAgentRatio == right.heavyAgentRatio
        && left.heavyWorkIterations == right.heavyWorkIterations
        && left.benchmarkFrames == right.benchmarkFrames;
}

void runSuite(
    const vector<BenchmarkConfig>& configs,
    vector<pair<BenchmarkConfig, BenchmarkResult>>& suiteResults
)
{
    BenchmarkConfig defaultBaselineConfig{};
    double defaultSequentialBaselineMs = 0.0;

    string heavyBaselineScheduler;
    size_t heavyBaselineAgentCount = 0;
    size_t heavyBaselineGrainSize = 0;
    float heavyBaselineRatio = 0.0f;
    int heavyBaselineIterations = 0;
    int heavyBaselineFrames = 0;
    double heavySchedulerBaselineMs = 0.0;

    for (const BenchmarkConfig& config : configs)
    {
        if (shouldSkipBenchmark(config))
        {
            cout << "[Benchmark] Skipping expensive combination: Suite=" << config.suiteName
                 << " Scheduler=" << config.schedulerName
                 << " Agents=" << config.agentCount
                 << " HeavyRatio=" << config.heavyAgentRatio
                 << " HeavyIterations=" << config.heavyWorkIterations << '\n';
            continue;
        }

        BenchmarkResult result = runBenchmark(config);
        double baselineMs = result.averageFrameTimeMs;

        if (config.suiteName == "heavy_parallel")
        {
            const bool isNewHeavyBaseline = config.workerCount == 1
                || config.schedulerName != heavyBaselineScheduler
                || config.agentCount != heavyBaselineAgentCount
                || config.grainSize != heavyBaselineGrainSize
                || config.heavyAgentRatio != heavyBaselineRatio
                || config.heavyWorkIterations != heavyBaselineIterations
                || config.benchmarkFrames != heavyBaselineFrames;

            if (isNewHeavyBaseline)
            {
                heavyBaselineScheduler = config.schedulerName;
                heavyBaselineAgentCount = config.agentCount;
                heavyBaselineGrainSize = config.grainSize;
                heavyBaselineRatio = config.heavyAgentRatio;
                heavyBaselineIterations = config.heavyWorkIterations;
                heavyBaselineFrames = config.benchmarkFrames;
                heavySchedulerBaselineMs = result.averageFrameTimeMs;
            }

            baselineMs = heavySchedulerBaselineMs;
        }
        else
        {
            if (config.schedulerName == "Sequential" || !sameDefaultScenario(config, defaultBaselineConfig))
            {
                defaultBaselineConfig = config;
                defaultSequentialBaselineMs = result.averageFrameTimeMs;
            }

            baselineMs = defaultSequentialBaselineMs;
        }

        applyDerivedMetrics(result, config, baselineMs);
        suiteResults.emplace_back(config, result);
    }
}

int main()
{
    constexpr bool runDefaultBenchmarks = false;
    constexpr bool runLightOverheadSuite = true;
    constexpr bool runHeavyParallelSuite = true;

    vector<BenchmarkConfig> defaultConfigs;
    vector<pair<BenchmarkConfig, BenchmarkResult>> defaultResults;
    vector<pair<BenchmarkConfig, BenchmarkResult>> lightOverheadResults;
    vector<pair<BenchmarkConfig, BenchmarkResult>> heavyParallelResults;

    if (runDefaultBenchmarks)
    {
        defaultConfigs = createDefaultSuite();
        defaultResults.reserve(defaultConfigs.size());
        runSuite(defaultConfigs, defaultResults);
    }

    if (runLightOverheadSuite)
    {
        vector<BenchmarkConfig> lightOverheadConfigs = createLightOverheadSuite();
        lightOverheadResults.reserve(lightOverheadConfigs.size());
        runSuite(lightOverheadConfigs, lightOverheadResults);
    }

    if (!lightOverheadResults.empty())
    {
        BenchmarkCSVExporter::exportExtendedResults("benchmark_results_light_overhead.csv", lightOverheadResults);
            cout << "[Benchmark] Exported benchmark_results_light_overhead.csv with " << lightOverheadResults.size() << " rows\n";
    }

    if (runHeavyParallelSuite)
    {
        vector<BenchmarkConfig> heavyParallelConfigs = createHeavyParallelSuite();
        heavyParallelResults.reserve(heavyParallelConfigs.size());
        runSuite(heavyParallelConfigs, heavyParallelResults);
    }

    // BenchmarkCSVExporter::exportResults("benchmark_results.csv", defaultResults);

    if (!heavyParallelResults.empty())
    {
        BenchmarkCSVExporter::exportExtendedResults("benchmark_results_heavy_parallel.csv", heavyParallelResults);
        cout << "[Benchmark] Exported benchmark_results_heavy_parallel.csv with " << heavyParallelResults.size() << " rows\n";
    }

    // cout << "[Benchmark] Exported benchmark_results.csv with " << defaultResults.size() << " rows\n";

    return 0;
}
