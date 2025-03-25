# tsp_solver.py

import time
import numpy as np
from itertools import permutations, combinations

def tsp_brute_force(distance_matrix, start='start', waypoints=None):
    start_time = time.time()

    if waypoints is None:
        waypoints = [wp for wp in distance_matrix.columns if wp != start]

    best_path = []
    min_total_distance = float('inf')
    best_path_distances = []

    for perm in permutations(waypoints):
        route = [start] + list(perm) + [start]
        total_dist = 0
        segment_distances = []

        for i in range(len(route) - 1):
            from_loc, to_loc = route[i], route[i + 1]
            dist = distance_matrix.loc[from_loc, to_loc]
            total_dist += dist
            segment_distances.append((from_loc, to_loc, round(dist, 2)))

        if total_dist < min_total_distance:
            min_total_distance = total_dist
            best_path = route
            best_path_distances = segment_distances

    elapsed = time.time() - start_time
    return best_path, min_total_distance, best_path_distances, elapsed


def tsp_held_karp(distance_matrix):
    start_time = time.time()

    all_points = list(distance_matrix.columns)
    n = len(all_points)
    point_idx = {name: i for i, name in enumerate(all_points)}
    idx_point = {i: name for name, i in point_idx.items()}

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = distance_matrix.loc[all_points[i], all_points[j]]

    dp = {}
    path_record = {}

    for k in range(1, n):
        dp[(1 << k, k)] = dist_matrix[0][k]
        path_record[(1 << k, k)] = 0

    for subset_size in range(2, n):
        for subset in combinations(range(1, n), subset_size):
            bits = sum(1 << x for x in subset)
            for k in subset:
                prev_bits = bits & ~(1 << k)
                min_dist = float('inf')
                prev_node = -1
                for m in subset:
                    if m == k:
                        continue
                    if (prev_bits, m) in dp:
                        new_dist = dp[(prev_bits, m)] + dist_matrix[m][k]
                        if new_dist < min_dist:
                            min_dist = new_dist
                            prev_node = m
                dp[(bits, k)] = min_dist
                path_record[(bits, k)] = prev_node

    full_bits = (1 << n) - 2
    min_cost = float('inf')
    last_node = -1

    for k in range(1, n):
        cost = dp[(full_bits, k)] + dist_matrix[k][0]
        if cost < min_cost:
            min_cost = cost
            last_node = k

    order = []
    bits = full_bits
    current = last_node

    while current != 0:
        order.append(current)
        prev = path_record[(bits, current)]
        bits = bits & ~(1 << current)
        current = prev

    order.append(0)
    order.reverse()

    optimal_path = [idx_point[i] for i in order] + [idx_point[0]]
    elapsed_time = time.time() - start_time
    return optimal_path, min_cost, elapsed_time
