#include "Agent.h"
#include "WorldSnapshot.h"

#include <algorithm>
#include <cmath>
#include <random>

using std::max;
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

void Agent::update(
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
)
{
    if (isHeavyAgent)
    {
        runHeavyWork(heavyWorkIterations);
    }

    const float toTargetX = targetX - x;
    const float toTargetY = targetY - y;
    const float targetDistance = sqrt(toTargetX * toTargetX + toTargetY * toTargetY);
    constexpr float epsilon = 0.1f;

    if (targetDistance <= epsilon)
    {
        assignRandomTarget(worldWidth, worldHeight);
        return;
    }

    const float normalizedTargetX = toTargetX / targetDistance;
    const float normalizedTargetY = toTargetY / targetDistance;

    const int neighborCount = snapshot.countNeighbors(agentIndex, perceptionRadius);
    const Vec2 avoidance = snapshot.computeAvoidanceVector(agentIndex, perceptionRadius);

    float desiredX = normalizedTargetX + avoidanceStrength * avoidance.x;
    float desiredY = normalizedTargetY + avoidanceStrength * avoidance.y;
    const float desiredLength = sqrt(desiredX * desiredX + desiredY * desiredY);

    if (desiredLength > 0.000001f)
    {
        desiredX /= desiredLength;
        desiredY /= desiredLength;
    }
    else
    {
        desiredX = normalizedTargetX;
        desiredY = normalizedTargetY;
    }

    const float slowdown = max(0.0f, densitySlowdown);
    const float speedFactor = 1.0f / (1.0f + slowdown * static_cast<float>(neighborCount));
    const float movement = speed * speedFactor * deltaTime;

    if (movement >= targetDistance)
    {
        x = targetX;
        y = targetY;
        assignRandomTarget(worldWidth, worldHeight);
        return;
    }

    x += desiredX * movement;
    y += desiredY * movement;
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
