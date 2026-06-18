#include "WorldSnapshot.h"

#include <algorithm>
#include <cmath>

using std::ceil;
using std::clamp;
using std::max;
using std::sqrt;

void WorldSnapshot::build(
    const vector<Agent>& agents,
    float worldWidth,
    float worldHeight,
    float cellSize
)
{
    width = max(worldWidth, 1.0f);
    height = max(worldHeight, 1.0f);
    cellExtent = max(cellSize, 0.001f);
    columns = max(1, static_cast<int>(ceil(width / cellExtent)));
    rows = max(1, static_cast<int>(ceil(height / cellExtent)));

    positions.clear();
    positions.reserve(agents.size());

    const size_t cellCount = static_cast<size_t>(columns) * static_cast<size_t>(rows);
    cells.clear();
    cells.resize(cellCount);

    for (size_t i = 0; i < agents.size(); ++i)
    {
        const Vec2 position{agents[i].getX(), agents[i].getY()};
        positions.push_back(position);
        cells[positionToCellIndex(position)].push_back(i);
    }
}

int WorldSnapshot::countNeighbors(size_t agentIndex, float radius) const
{
    if (agentIndex >= positions.size() || radius <= 0.0f)
    {
        return 0;
    }

    const Vec2 origin = positions[agentIndex];
    const int centerCellX = clampCellX(static_cast<int>(origin.x / cellExtent));
    const int centerCellY = clampCellY(static_cast<int>(origin.y / cellExtent));
    const float radiusSquared = radius * radius;
    int neighborCount = 0;

    // With cellSize == perceptionRadius, only the current and adjacent cells
    // can contain neighbors inside the query radius. This avoids O(N^2) scans.
    for (int y = max(0, centerCellY - 1); y <= clampCellY(centerCellY + 1); ++y)
    {
        for (int x = max(0, centerCellX - 1); x <= clampCellX(centerCellX + 1); ++x)
        {
            const vector<size_t>& cell = cells[static_cast<size_t>(y) * static_cast<size_t>(columns) + static_cast<size_t>(x)];
            for (const size_t otherIndex : cell)
            {
                if (otherIndex == agentIndex)
                {
                    continue;
                }

                const Vec2 other = positions[otherIndex];
                const float dx = origin.x - other.x;
                const float dy = origin.y - other.y;
                const float distanceSquared = dx * dx + dy * dy;

                if (distanceSquared <= radiusSquared)
                {
                    ++neighborCount;
                }
            }
        }
    }

    return neighborCount;
}

Vec2 WorldSnapshot::computeAvoidanceVector(size_t agentIndex, float radius) const
{
    if (agentIndex >= positions.size() || radius <= 0.0f)
    {
        return {};
    }

    const Vec2 origin = positions[agentIndex];
    const int centerCellX = clampCellX(static_cast<int>(origin.x / cellExtent));
    const int centerCellY = clampCellY(static_cast<int>(origin.y / cellExtent));
    const float radiusSquared = radius * radius;
    Vec2 avoidance{};

    for (int y = max(0, centerCellY - 1); y <= clampCellY(centerCellY + 1); ++y)
    {
        for (int x = max(0, centerCellX - 1); x <= clampCellX(centerCellX + 1); ++x)
        {
            const vector<size_t>& cell = cells[static_cast<size_t>(y) * static_cast<size_t>(columns) + static_cast<size_t>(x)];
            for (const size_t otherIndex : cell)
            {
                if (otherIndex == agentIndex)
                {
                    continue;
                }

                const Vec2 other = positions[otherIndex];
                const float dx = origin.x - other.x;
                const float dy = origin.y - other.y;
                const float distanceSquared = dx * dx + dy * dy;

                if (distanceSquared > 0.000001f && distanceSquared <= radiusSquared)
                {
                    const float distance = sqrt(distanceSquared);
                    const float weight = (radius - distance) / radius;
                    avoidance.x += (dx / distance) * weight;
                    avoidance.y += (dy / distance) * weight;
                }
            }
        }
    }

    return avoidance;
}

const Vec2& WorldSnapshot::getPosition(size_t agentIndex) const
{
    return positions[agentIndex];
}

size_t WorldSnapshot::size() const
{
    return positions.size();
}

size_t WorldSnapshot::positionToCellIndex(const Vec2& position) const
{
    const int cellX = clampCellX(static_cast<int>(position.x / cellExtent));
    const int cellY = clampCellY(static_cast<int>(position.y / cellExtent));
    return static_cast<size_t>(cellY) * static_cast<size_t>(columns) + static_cast<size_t>(cellX);
}

int WorldSnapshot::clampCellX(int cellX) const
{
    return clamp(cellX, 0, columns - 1);
}

int WorldSnapshot::clampCellY(int cellY) const
{
    return clamp(cellY, 0, rows - 1);
}
