#pragma once

#include <cstdint>
#include <random>

using std::mt19937;
using std::uint32_t;

// NPC agent in the simulation world.
class Agent
{
public:
    Agent(float x, float y, float targetX, float targetY, uint32_t randomSeed, bool isHeavyAgent);

    // Moves toward the target and assigns a new target when reached.
    void update(float deltaTime, float speed, float worldWidth, float worldHeight, int heavyWorkIterations);

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
