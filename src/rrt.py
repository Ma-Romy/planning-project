# rrt.py

import numpy as np
import random
import time

class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent

def rrt(grid, start, goal, step_size=10, max_iter=5000, goal_threshold=10, resolution=0.2):
    start_time = time.time()
    height, width = grid.shape
    start = tuple(reversed(start))
    goal = tuple(reversed(goal))
    tree = [Node(start)]
    visited = set()
    path = []

    def is_free(p):
        y, x = p
        return 0 <= y < height and 0 <= x < width and grid[y, x] == 255

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def steer(from_pos, to_pos):
        vec = np.array(to_pos) - np.array(from_pos)
        dist = np.linalg.norm(vec)
        if dist == 0:
            return from_pos
        vec = (vec / dist) * min(step_size, dist)
        return tuple((np.array(from_pos) + vec).astype(int))

    for _ in range(max_iter):
        x_rand = (random.randint(0, height - 1), random.randint(0, width - 1))
        nearest = min(tree, key=lambda node: distance(node.pos, x_rand))
        x_new = steer(nearest.pos, x_rand)
        if not is_free(x_new):
            continue
        new_node = Node(x_new, nearest)
        tree.append(new_node)
        visited.add(x_new)
        if distance(x_new, goal) < goal_threshold:
            goal_node = Node(goal, new_node)
            current = goal_node
            while current:
                path.append(current.pos)
                current = current.parent
            path.reverse()
            break

    total_distance = sum(distance(path[i - 1], path[i]) for i in range(1, len(path))) * resolution
    runtime = time.time() - start_time
    return path, total_distance, visited, runtime
