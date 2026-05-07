#include "Agent.h"

#include <cmath>
#include <random>

using std::sqrt;
using std::mt19937;
using std::sin;
using std::uniform_real_distribution;

Agent::Agent(float x, float y, float targetX, float targetY, uint32_t randomSeed, bool isHeavyAgent)
    : x(x),
      y(y),
      targetX(targetX),
      targetY(targetY),
      rng(randomSeed),
      isHeavyAgent(isHeavyAgent)
{
}

void Agent::update(float deltaTime, float speed, float worldWidth, float worldHeight, int heavyWorkIterations)
{
    if (isHeavyAgent)
    {
        runHeavyWork(heavyWorkIterations);
    }

    const float directionX = targetX - x;
    const float directionY = targetY - y;
    const float distance = sqrt(directionX * directionX + directionY * directionY);
    constexpr float epsilon = 0.1f;

    if (distance <= epsilon)
    {
        assignRandomTarget(worldWidth, worldHeight);
        return;
    }

    // Normalize the direction and move by speed * deltaTime.
    const float normalizedX = directionX / distance;
    const float normalizedY = directionY / distance;
    const float movement = speed * deltaTime;

    if (movement >= distance)
    {
        x = targetX;
        y = targetY;
        assignRandomTarget(worldWidth, worldHeight);
        return;
    }

    x += normalizedX * movement;
    y += normalizedY * movement;
}

void Agent::runHeavyWork(int iterations) const
{
    double accumulator = x * 0.001 + y * 0.002 + targetX * 0.003 + targetY * 0.004;

    // This deterministic CPU-bound loop represents expensive per-agent AI work
    for (int i = 0; i < iterations; ++i)
    {
        accumulator += sin(accumulator + static_cast<double>(i) * 0.001);
        accumulator *= 0.999999;
    }

    volatile double sink = accumulator;
    (void)sink;
}

void Agent::assignRandomTarget(float worldWidth, float worldHeight)
{
    uniform_real_distribution<float> xDistribution(0.0f, worldWidth);
    uniform_real_distribution<float> yDistribution(0.0f, worldHeight);

    targetX = xDistribution(rng);
    targetY = yDistribution(rng);
}

void Agent::setTarget(float newTargetX, float newTargetY)
{
    targetX = newTargetX;
    targetY = newTargetY;
}

bool Agent::isCloseToTarget(float epsilon) const
{
    const float directionX = targetX - x;
    const float directionY = targetY - y;
    const float distanceSquared = directionX * directionX + directionY * directionY;

    return distanceSquared <= epsilon * epsilon;
}

float Agent::getX() const
{
    return x;
}

float Agent::getY() const
{
    return y;
}

float Agent::getTargetX() const
{
    return targetX;
}

float Agent::getTargetY() const
{
    return targetY;
}

bool Agent::getIsHeavyAgent() const
{
    return isHeavyAgent;
}
