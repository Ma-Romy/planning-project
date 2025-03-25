# astar.py

import heapq
import time
import numpy as np

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(pos, grid):
    y, x = pos
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx] == 255:
            cost = 0.282 if dy != 0 and dx != 0 else 0.2
            neighbors.append(((ny, nx), cost))
    return neighbors

def astar(grid, start, goal, resolution=0.2):
    start_time = time.time()
    start = tuple(reversed(start))
    goal = tuple(reversed(goal))
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.add(current)

        if current == goal:
            break

        for neighbor, move_cost in get_neighbors(current, grid):
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
                came_from[neighbor] = current

    path = []
    current = goal
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    distance = sum(
        0.282 if abs(path[i][0] - path[i-1][0]) and abs(path[i][1] - path[i-1][1]) else 0.2
        for i in range(1, len(path))
    ) * resolution

    runtime = time.time() - start_time
    return path, distance, visited, runtime
