#include "BenchmarkCSVExporter.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

using std::ofstream;
using std::runtime_error;
using std::setprecision;

void BenchmarkCSVExporter::exportResults(
    const string& filename,
    const vector<pair<BenchmarkConfig, BenchmarkResult>>& results
)
{
    ofstream file(filename);

    if (!file)
    {
        throw runtime_error("Failed to open benchmark CSV file.");
    }

    file << setprecision(10);
    file << "Scheduler,Threads,Agents,GrainSize,HeavyRatio,HeavyIterations,Frames,"
         << "AvgFrameMs,MinFrameMs,MaxFrameMs,Speedup,Efficiency\n";

    for (const auto& [config, result] : results)
    {
        file << config.schedulerName << ','
             << config.workerCount << ','
             << config.agentCount << ','
             << config.grainSize << ','
             << config.heavyAgentRatio << ','
             << config.heavyWorkIterations << ','
             << config.benchmarkFrames << ','
             << result.averageFrameTimeMs << ','
             << result.minFrameTimeMs << ','
             << result.maxFrameTimeMs << ','
             << result.speedup << ','
             << result.efficiency << '\n';
    }
}

void BenchmarkCSVExporter::exportExtendedResults(
    const string& filename,
    const vector<pair<BenchmarkConfig, BenchmarkResult>>& results
)
{
    ofstream file(filename);

    if (!file)
    {
        throw runtime_error("Failed to open extended benchmark CSV file.");
    }

    file << setprecision(10);
    file << "Suite,Scheduler,Threads,Agents,GrainSize,HeavyRatio,HeavyIterations,HeavyLoad,Frames,"
         << "AvgFrameMs,MinFrameMs,MaxFrameMs,Speedup,Efficiency,ThroughputAgentsPerSec\n";

    for (const auto& [config, result] : results)
    {
        file << config.suiteName << ','
             << config.schedulerName << ','
             << config.workerCount << ','
             << config.agentCount << ','
             << config.grainSize << ','
             << config.heavyAgentRatio << ','
             << config.heavyWorkIterations << ','
             << result.heavyLoad << ','
             << config.benchmarkFrames << ','
             << result.averageFrameTimeMs << ','
             << result.minFrameTimeMs << ','
             << result.maxFrameTimeMs << ','
             << result.speedup << ','
             << result.efficiency << ','
             << result.throughputAgentsPerSec << '\n';
    }
}
