#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

using std::mt19937;
using std::size_t;
using std::uint32_t;

class WorldSnapshot;

// NPC agent in the simulation world.
class Agent
{
public:
    Agent(float x, float y, float targetX, float targetY, uint32_t randomSeed, bool isHeavyAgent);

    // Moves toward the target and assigns a new target when reached.
    void update(float deltaTime, float speed, float worldWidth, float worldHeight, int heavyWorkIterations);

    // Snapshot interaction update. The snapshot is read-only, and this method
    // only mutates this agent, which keeps parallel execution race-free.
    void update(
        float deltaTime,
        float speed,
        float worldWidth,
        float worldHeight,
        const WorldSnapshot& snapshot,
        size_t agentIndex,
        float perceptionRadius,
        float avoidanceStrength,
        float densitySlowdown,
        int heavyWorkIterations
    );

    // Assigns a new random target inside the given world bounds.
    void assignRandomTarget(float worldWidth, float worldHeight);

    void setTarget(float newTargetX, float newTargetY);
    bool isCloseToTarget(float epsilon) const;

    float getX() const;
    float getY() const;
    float getTargetX() const;
    float getTargetY() const;
    bool getIsHeavyAgent() const;

private:
    void runHeavyWork(int iterations) const;

    float x;
    float y;
    float targetX;
    float targetY;
    mt19937 rng;
    bool isHeavyAgent;
};
