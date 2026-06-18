#pragma once

#include "BenchmarkTypes.h"

#include <string>
#include <utility>
#include <vector>

using std::pair;
using std::string;
using std::vector;

// Writes all benchmark rows after execution finishes.
class BenchmarkCSVExporter
{
public:
    static void exportResults(
        const string& filename,
        const vector<pair<BenchmarkConfig, BenchmarkResult>>& results
    );

    static void exportExtendedResults(
        const string& filename,
        const vector<pair<BenchmarkConfig, BenchmarkResult>>& results
    );

    static void exportInteractionSnapshotResults(
        const string& filename,
        const vector<pair<BenchmarkConfig, BenchmarkResult>>& results
    );
};
