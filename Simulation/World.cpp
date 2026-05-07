#include "World.h"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>

using std::clamp;
using std::mt19937;
using std::pair;
using std::size_t;
using std::uniform_real_distribution;
using std::vector;

World::World(float width, float height)
    : width(width), height(height)
{
}

void World::initializeAgents(size_t agentCount, mt19937& rng, float heavyAgentRatio)
{
    agents.clear();
    agents.reserve(agentCount);

    const float clampedHeavyRatio = clamp(heavyAgentRatio, 0.0f, 1.0f);
    uniform_real_distribution<float> probabilityDistribution(0.0f, 1.0f);

    for (size_t i = 0; i < agentCount; ++i)
    {
        const auto position = generateRandomPosition(rng);
        const auto target = generateRandomPosition(rng);
        const auto agentSeed = rng();
        const bool isHeavyAgent = probabilityDistribution(rng) < clampedHeavyRatio;

        agents.emplace_back(position.first, position.second, target.first, target.second, agentSeed, isHeavyAgent);
    }
}

pair<float, float> World::generateRandomPosition(mt19937& rng) const
{
    uniform_real_distribution<float> xDistribution(0.0f, width);
    uniform_real_distribution<float> yDistribution(0.0f, height);

    return {xDistribution(rng), yDistribution(rng)};
}

float World::getWidth() const
{
    return width;
}

float World::getHeight() const
{
    return height;
}

vector<Agent>& World::getAgents()
{
    return agents;
}

const vector<Agent>& World::getAgents() const
{
    return agents;
}
