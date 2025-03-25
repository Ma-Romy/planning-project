# greedy.py

import heapq
import time
import numpy as np
from astar import heuristic, get_neighbors

def greedy_best_first(grid, start, goal, resolution=0.2):
    start_time = time.time()
    start = tuple(reversed(start))
    goal = tuple(reversed(goal))
    open_set = [(heuristic(start, goal), start)]
    came_from = {}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.add(current)
        if current == goal:
            break

        for neighbor, _ in get_neighbors(current, grid):
            if neighbor not in visited and neighbor not in came_from:
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from.get(current, start)
    path.append(start)
    path.reverse()

    distance = sum(
        0.282 if abs(path[i][0] - path[i-1][0]) and abs(path[i][1] - path[i-1][1]) else 0.2
        for i in range(1, len(path))
    ) * resolution

    runtime = time.time() - start_time
    return path, distance, visited, runtime
