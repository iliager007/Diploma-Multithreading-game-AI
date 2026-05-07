#pragma once

#include "Agent.h"

#include <cstddef>
#include <random>
#include <utility>
#include <vector>

using std::mt19937;
using std::pair;
using std::size_t;
using std::vector;

// Stores world dimensions and owns all NPC agents.
class World
{
public:
    World(float width, float height);

    // Creates agents and deterministically marks some as heavy.
    void initializeAgents(size_t agentCount, mt19937& rng, float heavyAgentRatio);

    pair<float, float> generateRandomPosition(mt19937& rng) const;

    float getWidth() const;
    float getHeight() const;
    vector<Agent>& getAgents();
    const vector<Agent>& getAgents() const;

private:
    float width;
    float height;
    vector<Agent> agents;
};
