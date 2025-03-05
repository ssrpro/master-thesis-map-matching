"""
From the SSPD structure, we precompute 1+epsilon approximation of Frechet distances on exponent map of a semi separated pair.
Then using graph match structure we can query 1+epsilon approximation of Frechet distance of any trajectory to all path of the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast
from tqdm import tqdm

from .utilities import Point, Interval, interval_mid, point_point_distance, \
    INFINITY, circle_ab_segment_intersection, range_closed, lerp, lerp_point, line_point_distance, in_interval, epsilon_leq, epsilon_equal
from .input_graph import Graph, GraphNeighbour
from .sspd import SSPD_Transit_Node, SSPD_Transit_InternalNode, SSPD_Transit_LeafNode, TransitPair, query_sspd, get_sspd_info
from .graph_match import VertexMatchStructure, query_vertex_match, GraphMatch, EdgeMatchStructure, query_edge_match
from .frechet_distance import optimize_frechet_distance_graph_segment, find_interval_in_free_space, critical_optimize_frechet_distance_polyline_segment

@dataclass(frozen=True)
class Epsilon_Frechet_Structure:
    """
    The epsilon Frechet distance approximation of a transit pair.
    The vertex_distance is the straightest path Frechet distance.
    The exponent structure map the exp index (dx, dy, i) to index j. Accessing (dx, dy, i) is done as dx * R + R + (dy * R + R) * DS + i * DS * DS,
    where (R, DS, I) = exponent_structure_size.
    One index j1 or j2 maps to a displaced point relative to the origin using index_to_point list.
    Two indices (j1, j2) has a precomputed distance matrix accessed by j1 + J * j2, where J = exponent_distance_matrix_row_and_column_size.
    For the exponent map, the interval is also defined as (alpha, beta).
    """
    vertex_distance: float

    epsilon: float
    exponent_structure_index_to_index: list[int | None]
    exponent_structure_interval: Interval
    exponent_structure_size: tuple[int, int, int]
    index_to_point: list[Point]
    exponent_distance_matrix: list[float | None]
    exponent_distance_matrix_row_and_column_size: int

@dataclass(frozen=True)
class Epsilon_Frechet_Exact:
    """When the straight path Frechet distance is zero. This is a valid degenerate case when a semi separated pair is connected directly with a single edge."""
    vertex_distance: float

Epsilon_Frechet = Epsilon_Frechet_Structure | Epsilon_Frechet_Exact
"""Datastructure to efficiently query 1+epsilon approximation of Frechet distances on a graph and segment."""

def compute_sspd_frechet(sspd: SSPD_Transit_Node, graph: Graph, graph_neighbour: GraphNeighbour, closeness: float, epsilon: float) -> SSPD_Transit_Node:
    """Compute the datastructure to efficiently query 1+epsilon approximation of graph and segment on a SSPD tree."""

    sspd_info = get_sspd_info(sspd)
    print(f"Need to compute Fréchet distance for {sspd_info.transit_pair_count}, oh no!")
    progress_bar = tqdm(total=sspd_info.transit_pair_count)

    def compute_sspd_frechet_recurse(node: SSPD_Transit_Node) -> SSPD_Transit_Node:
        match node:
            case SSPD_Transit_InternalNode():
                new_matrix_mod: list[SSPD_Transit_Node | None] = [None] * (node.matrix_NM[0] * node.matrix_NM[1])

                for i in range(node.matrix_NM[0]):
                    for j in range(node.matrix_NM[1]):
                        new_matrix_mod[i * node.matrix_NM[1] + j] = compute_sspd_frechet_recurse(node.matrix[i * node.matrix_NM[1] + j])

                new_matrix = cast(list[SSPD_Transit_Node], new_matrix_mod)
                return SSPD_Transit_InternalNode(
                    node.weight_class,
                    node.region,
                    node.recursive_structure,
                    node.matrix_NM,
                    new_matrix
                )
            case SSPD_Transit_LeafNode():
                new_transit_pair: list[TransitPair] = []

                for transit_pair in node.transit_pair:
                    data = compute_frechet_datastructure(transit_pair.sspd_point, transit_pair.transit_point)
                    new_transit_pair.append(TransitPair(transit_pair.transit_point, transit_pair.sspd_point, data))

                return SSPD_Transit_LeafNode(
                    node.weight_class,
                    node.region,
                    node.point_set,
                    node.transit_points,
                    new_transit_pair
                )

    def compute_frechet_datastructure(sspd_point: int, transit_point: int) -> Epsilon_Frechet | None:
        vertex_distance_interval, _ = optimize_frechet_distance_graph_segment(graph.vertices[sspd_point], graph.vertices[transit_point], sspd_point, transit_point, graph, graph_neighbour, closeness)
        vertex_distance = interval_mid(vertex_distance_interval)

        if vertex_distance is None:
            return None
        elif vertex_distance == 0:
            return Epsilon_Frechet_Exact(vertex_distance)
        else:
            point_p = graph.vertices[transit_point]
            point_q = graph.vertices[sspd_point]
            C1 = 4 * math.sqrt(2)
            L = vertex_distance
            exponent_structure_interval = Interval(epsilon * L * 0.5, L / epsilon)
            R = math.ceil(C1 / (epsilon * 2))
            DS = 2 * R + 1
            alpha, beta = exponent_structure_interval
            II = math.ceil(math.log2(beta / alpha))

            exponent_structure_index_to_index: list[int | None] = []
            index_to_point: list[Point] = []

            for i in range(II):
                for dy_i in range(0, DS):
                    for dx_i in range(0, DS):
                        # If i = 0, then fill everything.
                        # If i > 0, then ignore points with index i-1 or less.
                        dx = dx_i - R
                        dy = dy_i - R
                        x = dx * (2 ** (2 + i)) * alpha * (epsilon / C1)
                        y = dy * (2 ** (2 + i)) * alpha * (epsilon / C1)
                        r = math.sqrt(x ** 2 + y ** 2)

                        if i > 0:
                            if not epsilon_equal(r, 0):
                                index = dx_i + dy_i * DS + i * DS * DS
                                ixy = max(0, math.floor(math.log2(r / alpha)))

                                if ixy == index:
                                    exponent_structure_index_to_index.append(len(index_to_point))
                                    index_to_point.append(Point(x, y))
                                else:
                                    exponent_structure_index_to_index.append(None)
                            else:
                                exponent_structure_index_to_index.append(None)
                        else:
                            exponent_structure_index_to_index.append(len(index_to_point))
                            index_to_point.append(Point(x, y))

            J = len(index_to_point)
            exponent_distance_matrix: list[float | None] = []
            total_decision_counter = 0

            for j2 in range(J):
                for j1 in range(J):
                    pp_origin = index_to_point[j1]
                    qq_origin = index_to_point[j2]

                    pp = Point(pp_origin.x + point_p.x, pp_origin.y + point_p.y)
                    qq = Point(qq_origin.x + point_q.x, qq_origin.y + point_q.y)
                    interval, decision_counter = optimize_frechet_distance_graph_segment(pp, qq, transit_point, sspd_point, graph, graph_neighbour, closeness)
                    distance_pp_qq = interval_mid(interval)
                    exponent_distance_matrix.append(distance_pp_qq)
                    total_decision_counter += decision_counter

            progress_bar.update(1)
            return Epsilon_Frechet_Structure(
                vertex_distance,
                epsilon=epsilon,
                exponent_structure_index_to_index=exponent_structure_index_to_index,
                exponent_structure_interval=exponent_structure_interval,
                exponent_structure_size=(R, DS, II),
                index_to_point=index_to_point,
                exponent_distance_matrix=exponent_distance_matrix,
                exponent_distance_matrix_row_and_column_size=J,
            )

    return compute_sspd_frechet_recurse(sspd)

def query_frechet_sspd(graph: Graph, sspd: SSPD_Transit_Node, p1_index: int, p2_index: int, a: Point, b: Point, epsilon: float) -> float | None:
    """Solves the Frechet decision problem between any segment ab and path in the graph starting from u and ending to w."""

    def approximate_frechet_distance_with_segment(graph: Graph, sspd_leaf: SSPD_Transit_LeafNode, p1_index: int, p2_index: int, a: Point, b: Point) -> float:
        """The 3 approximation of the Frechet distance between any segment ab is calculated to any path in the graph starting at p1 and ending at p2."""

        dist_p1: dict[int, float] = dict()
        dist_p2: dict[int, float] = dict()
        dist_ww: dict[int, float] = dict()

        for transit_point in sspd_leaf.transit_points:
            u = graph.vertices[p1_index]
            w = graph.vertices[transit_point]
            v = graph.vertices[p2_index]
            distance = critical_optimize_frechet_distance_polyline_segment(a, b, [u, w, v])

            if distance is not None:
                dist_ww[transit_point] = distance
            else:
                dist_ww[transit_point] = INFINITY

            dist_p1[transit_point] = INFINITY
            dist_p2[transit_point] = INFINITY

        for transit_pair in sspd_leaf.transit_pair:
            if isinstance(transit_pair.data, (Epsilon_Frechet_Exact, Epsilon_Frechet_Structure)):
                if transit_pair.sspd_point == p1_index:
                    dist_p1[transit_pair.transit_point] = transit_pair.data.vertex_distance
                if transit_pair.sspd_point == p2_index:
                    dist_p2[transit_pair.transit_point] = transit_pair.data.vertex_distance

        actual_dist = INFINITY
        for transit_point in sspd_leaf.transit_points:
            actual_dist = min(actual_dist, max(dist_p1[transit_point], dist_p2[transit_point]) + dist_ww[transit_point])
        return actual_dist

    def query_frechet_exponent_structure(graph: Graph, transit_pair: TransitPair, p: Point, q: Point) -> float | None:
        """Query the Frechet distance from the exponent grid distance matrix on the transit pair."""

        structure = transit_pair.data

        assert structure is not None
        assert isinstance(structure, Epsilon_Frechet)

        match structure:
            case Epsilon_Frechet_Exact():
                r = max(point_point_distance(p, graph.vertices[transit_pair.transit_point]), point_point_distance(q, graph.vertices[transit_pair.sspd_point]))
                dist_pq = point_point_distance(p, q)
                dist_ts = point_point_distance(graph.vertices[transit_pair.transit_point], graph.vertices[transit_pair.sspd_point])

                if epsilon_equal(dist_pq, 0) and epsilon_equal(dist_ts, 0):
                    return point_point_distance(p, graph.vertices[transit_pair.transit_point])
                elif not epsilon_equal(dist_pq, 0):
                    return line_point_distance(p, q, graph.vertices[transit_pair.transit_point])
                elif not epsilon_equal(dist_ts, 0):
                    return line_point_distance(graph.vertices[transit_pair.transit_point], graph.vertices[transit_pair.sspd_point], p)
                elif epsilon_equal(r, 0):
                    return 0

                dist = critical_optimize_frechet_distance_polyline_segment(
                    p,
                    q,
                    [graph.vertices[transit_pair.transit_point],
                    graph.vertices[transit_pair.sspd_point]]
                )
                return dist
            case Epsilon_Frechet_Structure():
                L = structure.vertex_distance
                r = max(point_point_distance(p, graph.vertices[transit_pair.transit_point]), point_point_distance(q, graph.vertices[transit_pair.sspd_point]))

                if r < structure.exponent_structure_interval.left:
                    return L - r
                elif r > structure.exponent_structure_interval.right:
                    return r
                else:
                    transit_point = graph.vertices[transit_pair.transit_point]
                    sspd_point = graph.vertices[transit_pair.sspd_point]

                    pp_exp_index = get_exponent_structure_exp_index(p, transit_point, structure)
                    qq_exp_index = get_exponent_structure_exp_index(q, sspd_point, structure)

                    # TODO HANDLE CASE WHEN less than alpha!!!
                    pp_index = structure.exponent_structure_index_to_index[pp_exp_index]
                    qq_index = structure.exponent_structure_index_to_index[qq_exp_index]

                    # print(structure.exponent_structure_interval, " ::: ", structure.exponent_structure_size)
                    # print(structure.exponent_structure_index_to_index)
                    # print(pp_index, qq_index)

                    assert pp_index is not None and qq_index is not None

                    pp = structure.index_to_point[pp_index]
                    qq = structure.index_to_point[qq_index]

                    assert pp is not None and qq is not None, "Function get_exponent_structure_exp_index must return valid indices."

                    grid_cell_dist = max(point_point_distance(p, pp), point_point_distance(q, qq))

                    matrix_index = pp_index + qq_index * structure.exponent_distance_matrix_row_and_column_size
                    matrix_dist = structure.exponent_distance_matrix[matrix_index]

                    if matrix_dist is not None:
                        return matrix_dist - grid_cell_dist
                    else:
                        return None

    def get_exponent_structure_exp_index(p: Point, origin: Point, structure: Epsilon_Frechet_Structure) -> int:
        """Query the index of the exponent grid structure."""

        alpha = structure.exponent_structure_interval.left
        (R, DS, _) = structure.exponent_structure_size

        pp_radius = point_point_distance(p, origin)

        if not epsilon_equal(pp_radius, 0):
            pp_i = max(0, math.floor(math.log2(pp_radius / alpha)))
        else:
            pp_i = 0

        pp_dx = (p.x - origin.x) / (2 ** (2 + pp_i) * alpha)
        pp_dy = (p.y - origin.y) / (2 ** (2 + pp_i) * alpha)

        pp_dx_index = round(pp_dx * R + R)
        pp_dy_index = round(pp_dy * R + R)
        pp_index = pp_dx_index + pp_dy_index * DS + pp_i * DS * DS

        return pp_index

    p1 = graph.vertices[p1_index]
    p2 = graph.vertices[p2_index]

    sspd_leaf = query_sspd(sspd, p1, p2)
    r = approximate_frechet_distance_with_segment(graph, sspd_leaf, p1_index, p2_index, a, b)

    minimun_distance = INFINITY

    if r < INFINITY:
        for w_index in sspd_leaf.transit_points:
            w = graph.vertices[w_index]

            w_ball3r_intersection = circle_ab_segment_intersection(w, 3 * r, a, b)
            if w_ball3r_intersection is not None:
                at, bt = w_ball3r_intersection

                transit_pair_p1 = None
                transit_pair_p2 = None

                for pair in sspd_leaf.transit_pair:
                    if pair.sspd_point == p1_index and pair.transit_point == w_index:
                        transit_pair_p1 = pair
                    if pair.sspd_point == p2_index and pair.transit_point == w_index:
                        transit_pair_p2 = pair

                assert transit_pair_p1 is not None and transit_pair_p2 is not None, "Transit pairs must exist by construction"

                t_steps = math.ceil(1 / epsilon)
                for t_index in range_closed(0, t_steps):
                    tt = lerp(t_index / t_steps, at, bt)
                    t = lerp_point(a, b, tt)

                    dist_at = query_frechet_exponent_structure(graph, transit_pair_p1, t, a)
                    dist_tb = query_frechet_exponent_structure(graph, transit_pair_p2, t, b)

                    if dist_at is not None and dist_tb is not None:
                        dist_atb = max(dist_at, dist_tb)
                        minimun_distance = min(dist_atb, minimun_distance)

    return minimun_distance

def frechet_distance_stage_1(graph: Graph, sspd: SSPD_Transit_Node, p1_index: int, p2_index: int) -> float | None:
    """Solves the Frechet decision problem between uw and path in the graph starting from u and ending to w."""

    p1 = graph.vertices[p1_index]
    p2 = graph.vertices[p2_index]
    sspd_leaf = query_sspd(sspd, p1, p2)

    dist_p1: dict[int, float] = dict()
    dist_p2: dict[int, float] = dict()
    dist_ww: dict[int, float] = dict()

    for transit_point in sspd_leaf.transit_points:
        dist_ww[transit_point] = line_point_distance(p1, p2, graph.vertices[transit_point])
        dist_p1[transit_point] = INFINITY
        dist_p2[transit_point] = INFINITY

    for transit_pair in sspd_leaf.transit_pair:
        if transit_pair.data is not None:
            structure = cast(Epsilon_Frechet_Structure, transit_pair.data)

            if transit_pair.sspd_point == p1_index:
                dist_p1[transit_pair.transit_point] = structure.vertex_distance
            if transit_pair.sspd_point == p2_index:
                dist_p2[transit_pair.transit_point] = structure.vertex_distance

    actual_dist = INFINITY
    for transit_point in sspd_leaf.transit_points:
        current_dist = max(dist_p1[transit_point], dist_p2[transit_point]) + dist_ww[transit_point]
        if current_dist < actual_dist:
            actual_dist = current_dist

    return actual_dist

def decide_frechet_distance_stage_2(graph: Graph, graph_neighbour: GraphNeighbour, sspd: SSPD_Transit_Node, vertex_match: VertexMatchStructure, a: Point, b: Point, lamb: float, epsilon: float) -> int:
    """
    Solves the decision Frechet decision problem between any segment ab in any path as epsilon approximation.
    Return value:
        -1 means lamb is less than epsilon optimal, i.e. not feasible
         0 means lamb is within epsilon optimal range, i.e. strictly feasible
         1 means lamb is greater than epsilon optimal, i.e. feasible
    """

    Ta = query_vertex_match(graph, graph_neighbour, vertex_match, a, lamb)
    Tb = query_vertex_match(graph, graph_neighbour, vertex_match, b, lamb)

    if len(Ta) == 0 or len(Tb) == 0:
        return -1

    lamb_approx = INFINITY

    for _, ui, _ in Ta:
        for _, vi, _ in Tb:
            u_index = vertex_match.kcenter_list[ui].vertex
            v_index = vertex_match.kcenter_list[vi].vertex
            ruv = query_frechet_sspd(graph, sspd, u_index, v_index, a, b, epsilon)

            if ruv is not None:
                lamb_approx = min(lamb_approx, ruv)

    epsilon = vertex_match.epsilon
    optimal_bound = Interval(lamb, ((1 + epsilon) ** 2) * lamb)

    if in_interval(lamb_approx, optimal_bound):
        return 0
    elif lamb_approx < optimal_bound.left:
        return -1
    else: # lamb_approx > optimal_bound.right:
        return 1

@dataclass(frozen=True)
class TrajectoryGraphEdge:
    capacity: float
    edge_1_vertex_index: int
    edge_2_vertex_index: int
    next_node: TrajectoryGraphNode

@dataclass(frozen=True)
class TrajectoryGraphNode:
    trajectory_index: int
    match_index: int
    children: list[TrajectoryGraphEdge]
    matched_point: Point

@dataclass(frozen=True)
class TrajectoryGraph:
    node_in_trajectory_index: list[dict[int, TrajectoryGraphNode]]

def build_trajectory_graph(graph: Graph, sspd: SSPD_Transit_Node, qs: list[Point], Tqs: list[list[GraphMatch]], lamb: float, epsilon: float) -> TrajectoryGraph:
    trajectory_graph = TrajectoryGraph([])

    for _ in range(len(qs)):
        trajectory_graph.node_in_trajectory_index.append(dict())

    for i in range(len(qs) - 1):
        ai = qs[i]
        ak = qs[i+1]
        for bi_index, (bi, ci_index, di_index) in enumerate(Tqs[i]):
            for bk_index, (bk, ck_index, dk_index) in enumerate(Tqs[i+1]):
                min_capacity = INFINITY
                min_e12: tuple[int, int] | None = None

                for e1_index, e2_index in [(ci_index, ck_index), (ci_index, dk_index), (di_index, ck_index), (di_index, dk_index)]:
                    e1 = graph.vertices[e1_index]
                    e2 = graph.vertices[e2_index]
                    aip_interval = find_interval_in_free_space(e1, ai, ak, lamb)
                    akp_interval = find_interval_in_free_space(e2, ai, ak, lamb)

                    if aip_interval is not None and akp_interval is not None:
                        aip = lerp_point(ai, ak, aip_interval[0])
                        akp = lerp_point(ai, ak, akp_interval[1])
                        capacity = query_frechet_sspd(graph, sspd, e1_index, e2_index, aip, akp, epsilon)
                        if capacity is not None:
                            if capacity < min_capacity:
                                min_capacity = capacity
                                min_e12 = (e1_index, e2_index)

                if min_capacity < INFINITY:
                    assert min_e12 is not None, "Non infinity capacity should have a defined min_e12."

                    if bi_index not in trajectory_graph.node_in_trajectory_index[i]:
                        trajectory_graph.node_in_trajectory_index[i][bi_index] = TrajectoryGraphNode(i, bi_index, [], bi)
                    
                    if bk_index not in trajectory_graph.node_in_trajectory_index[i+1]:
                        trajectory_graph.node_in_trajectory_index[i+1][bk_index] = TrajectoryGraphNode(i+1, bk_index, [], bk)

                    bi_node = trajectory_graph.node_in_trajectory_index[i][bi_index]
                    bk_node = trajectory_graph.node_in_trajectory_index[i+1][bk_index]
                    bi_node.children.append(TrajectoryGraphEdge(min_capacity, min_e12[0], min_e12[1], bk_node))

    return trajectory_graph

def decide_lamb_flow_exists_trajectory_graph_decide(trajectory_graph: TrajectoryGraph, trajectory_length: int, lamb: float) -> bool:
    def decide_lamb_flow_exists_trajectory_graph_decide_recurse(node: TrajectoryGraphNode) -> bool:
        if node.trajectory_index == trajectory_length - 1:
            return True

        best_is = INFINITY
        for edge in node.children:
            if epsilon_leq(edge.capacity, lamb):
                exists = decide_lamb_flow_exists_trajectory_graph_decide_recurse(edge.next_node)

                if exists:
                    return True
                
                best_is = min(best_is, edge.capacity)

        return False

    for root in trajectory_graph.node_in_trajectory_index[0].values():
        exists = decide_lamb_flow_exists_trajectory_graph_decide_recurse(root)
        if exists:
            return True

    return False

def find_path_lamb_flow_exists_trajectory_graph(trajectory_graph: TrajectoryGraph, lamb: float) -> list[int] | None:
    def find_path_lamb_flow_exists_trajectory_graph_recurse(node: TrajectoryGraphNode) -> list[int] | None:
        for edge in node.children:
            if epsilon_leq(edge.capacity, lamb):
                child_path = find_path_lamb_flow_exists_trajectory_graph_recurse(edge.next_node)

                if child_path is not None:
                    return [node.match_index, *child_path]

        return None

    for root in trajectory_graph.node_in_trajectory_index[0].values():
        path = find_path_lamb_flow_exists_trajectory_graph_recurse(root)
        if path is not None:
            return path

    return None

def decide_frechet_distance_stage_3(graph: Graph, graph_neighbour: GraphNeighbour, sspd: SSPD_Transit_Node, edge_match: EdgeMatchStructure, qs: list[Point], epsilon: float, lamb: float) -> int:
    """
    Solves the decision Frechet decision problem between any trajectory a1a2...aq in any path as epsilon approximation.
    Return value:
        -1 means lamb is less than epsilon optimal, i.e. not feasible
         0 means lamb is within epsilon optimal range, i.e. strictly feasible
         1 means lamb is greater than epsilon optimal, i.e. feasible
    """

    Tqs: list[list[GraphMatch]] = []

    for a in qs:
        Tqs.append(query_edge_match(graph, graph_neighbour, edge_match, a, lamb)[0])

    trajectory_graph = build_trajectory_graph(graph, sspd, qs, Tqs, lamb, epsilon)
    if decide_lamb_flow_exists_trajectory_graph_decide(trajectory_graph, len(qs), lamb):
        return 1
    else:
        if decide_lamb_flow_exists_trajectory_graph_decide(trajectory_graph, len(qs), lamb * (1 + epsilon) ** 2):
            return 0
        else:
            return -1

def optimize_frechet_distance_stage_3(graph: Graph, graph_neighbour: GraphNeighbour, sspd: SSPD_Transit_Node, edge_match: EdgeMatchStructure, qs: list[Point], epsilon: float, closeness: float, boundary_range: Interval = Interval(0.0001, 10000)) -> Interval | None:
    """
    Solves the decision Frechet decision problem between any trajectory a1a2...aq in any path as epsilon approximation.
    Return value:
        -1 means lamb is less than epsilon optimal, i.e. not feasible
         0 means lamb is within epsilon optimal range, i.e. strictly feasible
         1 means lamb is greater than epsilon optimal, i.e. feasible
    """

    boundary_left = decide_frechet_distance_stage_3(graph, graph_neighbour, sspd, edge_match, qs, epsilon, boundary_range.left)
    boundary_right = decide_frechet_distance_stage_3(graph, graph_neighbour, sspd, edge_match, qs, epsilon, boundary_range.right)

    if boundary_left == 1:
        return Interval(0, 0)
    if boundary_right == -1:
        return None

    interval = boundary_range

    while interval.right - interval.left > closeness:
        sample = 0.5 * (interval.left + interval.right)
        decision = decide_frechet_distance_stage_3(graph, graph_neighbour, sspd, edge_match, qs, epsilon, sample)
        if decision == 1:
            interval = Interval(interval.left, sample)
        elif decision == -1:
            interval = Interval(sample, interval.right)
        else:
            return Interval(sample, sample)

    return interval
