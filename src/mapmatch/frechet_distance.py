from __future__ import annotations

import heapq

from .utilities import Point, Interval, solve_quadratic_equation, interval_intersect, point_point_distance, epsilon_leq, interval_valid, line_point_distance, lerp_point, line_mid_point, line_line_intersection, in_interval, epsilon_equal
from .input_graph import Graph, GraphNeighbour, GraphPath

def find_interval_in_free_space(p: Point, a: Point, b: Point, lamb: float) -> Interval | None:
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
    if roots is not None:
        return interval_intersect(Interval(min(roots[0], roots[1]), max(roots[0], roots[1])), Interval(0, 1))
    else:
        return None

def decide_frechet_distance_polyline_segment(a: Point, b: Point, polyline: list[Point], lamb: float) -> bool:
    """
    Solves the Frechet decision problem between any segment ab and polyline p1 ... pq.
    Return value:
        False means lamb is less than optimal, i.e. not feasible
        True  means lamb is greater than optimal, i.e. feasible
    """

    # Boundary condition
    if epsilon_leq(point_point_distance(polyline[0], a), lamb) and epsilon_leq(point_point_distance(polyline[-1], b), lamb):
        interval = Interval(0, 1)

        for p in polyline:
            free_space_interval = interval_intersect(find_interval_in_free_space(p, a, b, lamb), Interval(0, 1))

            if free_space_interval is None:
                return False
            
            interval = Interval(max(free_space_interval.left, interval.left), free_space_interval.right)
            if not interval_valid(interval):
                return False

        return interval_intersect(interval, Interval(1, 1)) is not None
    else:
        return False

def optimize_frechet_distance_polyline_segment(a: Point, b: Point, polyline: list[Point], closeness: float, boundary_range: Interval = Interval(0.001, 10000)) -> Interval | None:
    """
    Calculates the Frechet distance of a polysegment and segment by binary searching over the Frechet decision problem.
    Returns the interval (iF, iT) where
    - iF is the largest lambda found that is infeasible.
    - iT is the smallest lambda found that is feasible.
    The optimal Frechet distance lies within the interval (iF, iT).
    """
    boundary_left = decide_frechet_distance_polyline_segment(a, b, polyline, boundary_range.left)
    boundary_right = decide_frechet_distance_polyline_segment(a, b, polyline, boundary_range.right)

    if boundary_left:
        return Interval(0, 0)
    if not boundary_right:
        return None

    interval = boundary_range

    while interval.right - interval.left > closeness:
        sample = 0.5 * (interval.left + interval.right)
        decision = decide_frechet_distance_polyline_segment(a, b, polyline, sample)
        if decision:
            interval = Interval(interval.left, sample)
        else:
            interval = Interval(sample, interval.right)

    return interval

def critical_optimize_frechet_distance_polyline_segment(a: Point, b: Point, polyline: list[Point]) -> float | None:
    lambs: list[float] = []

    lambs.append(point_point_distance(a, polyline[0]))
    lambs.append(point_point_distance(a, polyline[-1]))

    for p in polyline:

        lambs.append(line_point_distance(a, b, p))

    for i in range(len(polyline)):
        for j in range(i + 1, len(polyline)):
            qi = polyline[i]
            qj = polyline[j]
            qij = line_mid_point(qi, qj)
            nij = Point(-(qj.y - qij.y) + qij.x, qj.x - qij.x + qij.y)
            t = line_line_intersection(a, b, qij, nij)
            if t is not None and in_interval(t[0], Interval(0, 1)):
                c = lerp_point(a, b, t[0])
                di = point_point_distance(qi, c)
                dj = point_point_distance(qj, c)
                assert epsilon_equal(di, dj)
                lambs.append(di)

    lambs_sorted = sorted(lambs)
    for lamb in lambs_sorted:
        if decide_frechet_distance_polyline_segment(a, b, polyline, lamb):
            return lamb
        
    return None


def decide_frechet_distance_graph_segment(a: Point, b: Point, u_index: int, w_index: int, graph: Graph, graph_neighbour: GraphNeighbour, lamb: float) -> bool:
    """
    Solves the Frechet decision problem between any segment ab and path in the graph starting from u and ending to w for a given lambda, where u, v are graph vertices.
    Return value:
        False means lamb is less than optimal, i.e. not feasible
        True  means lamb is greater than optimal, i.e. feasible
    """
    # Boundary condition
    if epsilon_leq(point_point_distance(graph.vertices[u_index], a), lamb) and epsilon_leq(point_point_distance(graph.vertices[w_index], b), lamb):
        dijkstra_queue: list[tuple[float, int]] = []
        visited: set[int] = set()
        heapq.heappush(dijkstra_queue, (0.0, u_index))

        while len(dijkstra_queue) > 0:
            i_priority, i_index = heapq.heappop(dijkstra_queue)

            if i_index not in visited:
                visited.add(i_index)

                for j_index in graph_neighbour.get_vertex_neighbours(i_index):
                    if j_index != w_index:
                        j_point = graph.vertices[j_index]
                        j_interval = interval_intersect(Interval(i_priority, 1), find_interval_in_free_space(j_point, a, b, lamb))

                        if j_interval is not None:
                            j_priority = j_interval.left

                            if j_index not in visited:
                                heapq.heappush(dijkstra_queue, (j_priority, j_index))
                        else:
                            continue
                    else:
                        return True
        return False
    else:
        return False

def optimize_frechet_distance_graph_segment(a: Point, b: Point, u_index: int, w_index: int, graph: Graph, graph_neighbour: GraphNeighbour, closeness: float, boundary_range: Interval = Interval(0.001, 100)) -> tuple[Interval | None, int]:
    """
    Calculates the Frechet distance of a graph and segment by binary searching over the Frechet decision problem.
    Returns the interval (iF, iT) where
    - iF is the largest lambda found that is infeasible.
    - iT is the smallest lambda found that is feasible.
    The optimal Frechet distance lies within the interval (iF, iT).
    """
    boundary_left = decide_frechet_distance_graph_segment(a, b, u_index, w_index, graph, graph_neighbour, boundary_range.left)
    boundary_right = decide_frechet_distance_graph_segment(a, b, u_index, w_index, graph, graph_neighbour, boundary_range.right)
    decision_counter = 0

    if boundary_left:
        return Interval(0, 0), 0
    if not boundary_right:
        return None, 0

    interval = boundary_range

    while interval.right - interval.left > closeness:
        sample = 0.5 * (interval.left + interval.right)
        decision = decide_frechet_distance_graph_segment(a, b, u_index, w_index, graph, graph_neighbour, sample)
        decision_counter += 1
        if decision:
            interval = Interval(interval.left, sample)
        else:
            interval = Interval(sample, interval.right)

    return interval, decision_counter

def find_path_frechet_distance_graph_segment(a: Point, b: Point, u_index: int, w_index: int, graph: Graph, graph_neighbour: GraphNeighbour, lamb: float) -> GraphPath | None:
    """
    Solves the Frechet decision problem between any segment ab and path in the graph starting from u and ending to w for a given lambda.
    Return value: a path that satisfies the condition.
    """

    # Boundary condition
    if epsilon_leq(point_point_distance(graph.vertices[u_index], a), lamb) and epsilon_leq(point_point_distance(graph.vertices[w_index], b), lamb):
        dijkstra_queue: list[tuple[float, int, GraphPath]] = []
        visited: set[int] = set()
        heapq.heappush(dijkstra_queue, (0, u_index, GraphPath(None, u_index)))

        while len(dijkstra_queue) > 0:
            i_priority, i_index, prev_path = heapq.heappop(dijkstra_queue)

            if i_index not in visited:
                visited.add(i_index)

                for j_index in graph_neighbour.neighbours[i_index]:
                    if j_index != w_index:
                        j_point = graph.vertices[j_index]
                        j_interval = interval_intersect(Interval(i_priority, 1), find_interval_in_free_space(j_point, a, b, lamb))

                        if j_interval is not None:
                            j_priority = j_interval.left

                            if j_index not in visited:
                                heapq.heappush(dijkstra_queue, (j_priority, j_index, GraphPath(prev_path, j_index)))
                        else:
                            continue
                    else:
                        return GraphPath(prev_path, j_index)
        return None
    else:
        return None
