#pragma once

enum class InteractionMode
{
    None,
    SnapshotNeighbors
};

inline const char* toString(InteractionMode mode)
{
    switch (mode)
    {
    case InteractionMode::None:
        return "None";
    case InteractionMode::SnapshotNeighbors:
        return "SnapshotNeighbors";
    }

    return "Unknown";
}
