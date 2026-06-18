#pragma once

#include "Agent.h"

#include <cstddef>
#include <vector>

using std::size_t;
using std::vector;

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

// Immutable frame state used for snapshot-based inter-agent queries.
// It stores copied positions, never references to live Agent objects, so
// worker threads can read it concurrently while writing only their own agents.
class WorldSnapshot
{
public:
    void build(
        const vector<Agent>& agents,
        float worldWidth,
        float worldHeight,
        float cellSize
    );

    int countNeighbors(size_t agentIndex, float radius) const;
    Vec2 computeAvoidanceVector(size_t agentIndex, float radius) const;

    const Vec2& getPosition(size_t agentIndex) const;
    size_t size() const;

private:
    size_t positionToCellIndex(const Vec2& position) const;
    int clampCellX(int cellX) const;
    int clampCellY(int cellY) const;

    vector<Vec2> positions;
    vector<vector<size_t>> cells;
    float width = 0.0f;
    float height = 0.0f;
    float cellExtent = 1.0f;
    int columns = 1;
    int rows = 1;
};
