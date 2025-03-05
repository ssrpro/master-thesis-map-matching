from __future__ import annotations

from numba import jit, prange, int64, float32 # type: ignore

import heapq

import math
from typing import cast, NamedTuple, Any
from tqdm import tqdm
import numpy as np

from .utilities import Point, Interval, INFINITY, EPSILON
from .input_graph import Graph, GraphNeighbour
from .sspd import SSPD_Transit_Node, SSPD_Transit_InternalNode, SSPD_Transit_LeafNode, TransitPair
from .frechet_distance import optimize_frechet_distance_graph_segment

@jit(nopython=True, fastmath=True)  # type: ignore
def interval_intersect(interval1: Interval, interval2: Interval, e: float = EPSILON) -> Interval:
    """Returns the intersection between two intervals where either of them may be None."""

    result = Interval(max(interval1.left, interval2.left), min(interval1.right, interval2.right))
    if interval_valid(result):
        return result
    else:
        return Interval(1.0, 0.0)

@jit(nopython=True, fastmath=True)  # type: ignore
def solve_quadratic_equation(A: float, B: float, C: float) -> Interval:
    """Solve quadratic equation Axx + Bx + C = 0 as a pair. If there are no roots, then None will be returned."""
    D = B ** 2 - 4 * A * C

    if D >= 0 and not epsilon_equal(A, 0):
        DD = math.sqrt(D)
        r1, r2 = ((-B - DD) / (2 * A), (-B + DD) / (2 * A))
        return Interval(min(r1, r2), max(r1, r2))
    else:
        return Interval(1.0, 0.0)

@jit(nopython=True, fastmath=True)  # type: ignore
def epsilon_equal(x: float, y: float, e: float=EPSILON):
    """Check whether x and y are equal within epsilon range."""
    return abs(x - y) < e

@jit(nopython=True, fastmath=True)  # type: ignore
def find_interval_in_free_space(p: Point, a: Point, b: Point, lamb: float) -> Interval:
    """
    Find the free space of a vertex p and line segment ab
    solve u.x(Y)^2 + v.y(Y)^2 = c^2 for (p.o - a.o) + -(b.o - a.o) Y
    """
    Ax = (b.x - a.x) ** 2
    Bx = 2 * (p.x - a.x) * -(b.x - a.x)
    Cx = (p.x - a.x) ** 2

    Ay = (b.y - a.y) ** 2
    By = 2 * (p.y - a.y) * -(b.y - a.y)
    Cy = (p.y - a.y) ** 2

    roots = solve_quadratic_equation(Ax + Ay, Bx + By, Cx + Cy - lamb ** 2)
    if interval_valid(roots):
        return interval_intersect(Interval(min(roots[0], roots[1]), max(roots[0], roots[1])), Interval(0, 1))
    else:
        return Interval(1.0, 0.0)

@jit(nopython=True, fastmath=True)  # type: ignore
def epsilon_leq(x: float, y: float, e: float=EPSILON):
    """Check whether x <= y, where equality is within epsilon range."""
    return x < y or epsilon_equal(x, y, e)

@jit(nopython=True, fastmath=True)  # type: ignore
def interval_valid(interval: Interval, e: float = EPSILON) -> bool:
    """Checks for interval [a, b] whether a <= b with EPSILON relaxation."""
    return interval.left - e < interval.right + e

@jit(nopython=True, fastmath=True)  # type: ignore
def interval_mid(interval: Interval) -> float:
    """Return the mid value of an interval."""
    return 0.5 * (interval.left + interval.right)

@jit(nopython=True, fastmath=True)  # type: ignore
def decide_frechet_distance_graph_segment_fast(a: Point, b: Point, u_index: int, w_index: int, graph: FastGraph, lamb: float) -> bool:
    """
    Solves the Frechet decision problem between any segment ab and path in the graph starting from u and ending to w for a given lambda, where u, v are graph vertices.
    Return value:
        False means lamb is less than optimal, i.e. not feasible
        True  means lamb is greater than optimal, i.e. feasible
    """
    # Boundary condition
    if epsilon_leq((graph.vertex_points[u_index * 2 + 0] - a.x) ** 2 + (graph.vertex_points[u_index * 2 + 1] - a.y) ** 2, lamb) and epsilon_leq((graph.vertex_points[w_index * 2 + 0] - b.x) ** 2 + (graph.vertex_points[w_index * 2 + 1] - a.y) ** 2, lamb):
        dijkstra_queue = [(float32(x), np.int32(x)) for x in range(0)]
        visited: set[int] = set()
        heapq.heappush(dijkstra_queue, (float32(0.0), np.int32(u_index)))

        while len(dijkstra_queue) > 0:
            i_priority, i_index = heapq.heappop(dijkstra_queue)

            if i_index not in visited:
                visited.add(i_index)

                neighbour_offset = graph.vertex_index_to_neighbours_offset[i_index]
                neighbour_size = graph.vertex_index_to_neighbours_size[i_index]
                for neighbour_index in range(neighbour_offset, neighbour_offset + neighbour_size):
                    j_index = graph.vertex_neighbours[neighbour_index]
                    if j_index != w_index:
                        j_point = Point(graph.vertex_points[j_index * 2 + 0], graph.vertex_points[j_index * 2 + 1])
                        j_interval = interval_intersect(Interval(i_priority, 1), find_interval_in_free_space(j_point, a, b, lamb))

                        if interval_valid(j_interval):
                            j_priority = j_interval.left

                            if j_index not in visited:
                                heapq.heappush(dijkstra_queue, (float32(j_priority), j_index))
                        else:
                            continue
                    else:
                        return True
        return False
    else:
        return False

@jit(nopython=True, nogil=True, parallel=True, fastmath=True)  # type: ignore
def compute_sspd_frechet_fill_fast(graph: FastGraph, transit_pair_list_count: int, transit_pair_list: Any, sparse_index_to_point_vector: Any, sparse_index_matrix: Any, closeness: float, boundary_range: Interval, progress_proxy):
    for transit_pair_index in range(transit_pair_list_count):
        sspd_point = transit_pair_list[transit_pair_index * 12 + 0]
        transit_point = transit_pair_list[transit_pair_index * 12 + 1]
        offset_sparse_index_to_point_vector = transit_pair_list[transit_pair_index * 12 + 7]
        offset_sparse_index_matrix = transit_pair_list[transit_pair_index * 12 + 8]
        size_sparse_index_to_point_vector = transit_pair_list[transit_pair_index * 12 + 10]

        for i in prange(size_sparse_index_to_point_vector):
            for j in prange(size_sparse_index_to_point_vector):
                pi = Point(sparse_index_to_point_vector[(offset_sparse_index_to_point_vector + i) * 2 + 0], sparse_index_to_point_vector[(offset_sparse_index_to_point_vector + i) * 2 + 1])
                pj = Point(sparse_index_to_point_vector[(offset_sparse_index_to_point_vector + j) * 2 + 0], sparse_index_to_point_vector[(offset_sparse_index_to_point_vector + j) * 2 + 1])

                boundary_left = decide_frechet_distance_graph_segment_fast(pi, pj, transit_point, sspd_point, graph, boundary_range.left)
                boundary_right = decide_frechet_distance_graph_segment_fast(pi, pj, transit_point, sspd_point, graph, boundary_range.right)

                if boundary_left:
                    sparse_index_matrix[offset_sparse_index_matrix + i * size_sparse_index_to_point_vector] = 0.0
                if not boundary_right:
                    sparse_index_matrix[offset_sparse_index_matrix + i * size_sparse_index_to_point_vector] = INFINITY

                interval = boundary_range

                while interval.right - interval.left > closeness:
                    sample = 0.5 * (interval.left + interval.right)
                    decision = decide_frechet_distance_graph_segment_fast(pi, pj, transit_point, sspd_point, graph, sample)
                    if decision:
                        interval = Interval(interval.left, sample)
                    else:
                        interval = Interval(sample, interval.right)

                if interval_valid(interval):
                    distance_pp_qq = interval_mid(interval)
                    sparse_index_matrix[offset_sparse_index_matrix + i * size_sparse_index_to_point_vector] = distance_pp_qq
                else:
                    sparse_index_matrix[offset_sparse_index_matrix + i * size_sparse_index_to_point_vector] = INFINITY

                progress_proxy.update(1)
