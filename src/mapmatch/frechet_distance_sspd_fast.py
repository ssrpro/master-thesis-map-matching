"""
From the SSPD structure, we precompute 1+epsilon approximation of Frechet distances on exponent map of a semi separated pair.
Then using graph match structure we can query 1+epsilon approximation of Frechet distance of any trajectory to all path of the graph.
"""

from __future__ import annotations

import math
from typing import cast, NamedTuple, Any
from tqdm import tqdm
import numpy as np

from .utilities import Point, Interval, INFINITY, interval_mid, epsilon_equal
from .input_graph import Graph, GraphNeighbour
from .sspd import SSPD_Transit_Node, SSPD_Transit_InternalNode, SSPD_Transit_LeafNode, TransitPair
from .frechet_distance import optimize_frechet_distance_graph_segment
from .frechet_distance_sspd import Epsilon_Frechet_Structure, Epsilon_Frechet_Exact

class TransitPairData(NamedTuple):
    # Vertex index of sspd point of the transit pair.
    sspd_point: int
    # Vertex index of transit point of the transit pair.
    transit_point: int
    # Number of exponential grid levels where level ranges from [\alpha, \beta]
    exp_level_count: int
    # Number of exponential grid rows or columns from zero to the end.
    exp_grid_radius_count: int
    # Number of exponential grid rows or columns. Total number of grids for each level will be that number squared.
    exp_grid_side_count: int
    # Matrix row/column count
    matrix_dimension: int
    # Offset of exponent_structure_index_to_sparse_index_vector
    offset_exponent_structure_index_to_sparse_index_vector: int
    # Offset of sparse_index_to_point_vector
    offset_sparse_index_to_point_vector: int
    # Offset of sparse_index_matrix
    offset_sparse_index_matrix: int
    # Size of exponent_structure_index_to_sparse_index_vector
    size_exponent_structure_index_to_sparse_index_vector: int
    # Size of sparse_index_to_point_vector
    size_sparse_index_to_point_vector: int
    # Size of sparse_index_matrix
    size_sparse_index_matrix: int

def transit_pair_data_get(transit_pairs: list[TransitPairData]) -> Any:
    result = np.zeros(len(transit_pairs) * 12, dtype=np.int64)

    for i, transit_pair in enumerate(transit_pairs):
        for x in transit_pair:
            assert x is not None

        result[i * 12 + 0] = np.int64(transit_pair[0])
        result[i * 12 + 1] = np.int64(transit_pair[1])
        result[i * 12 + 2] = np.int64(transit_pair[2])
        result[i * 12 + 3] = np.int64(transit_pair[3])
        result[i * 12 + 4] = np.int64(transit_pair[4])
        result[i * 12 + 5] = np.int64(transit_pair[5])
        result[i * 12 + 6] = np.int64(transit_pair[6])
        result[i * 12 + 7] = np.int64(transit_pair[7])
        result[i * 12 + 8] = np.int64(transit_pair[8])
        result[i * 12 + 9] = np.int64(transit_pair[9])
        result[i * 12 + 10] = np.int64(transit_pair[10])
        result[i * 12 + 11] = np.int64(transit_pair[11])

    return result

class TransitPairData2(NamedTuple):
    # Minimum level
    exp_alpha: float
    # Maxmimum level value
    exp_beta: float

class FastGraph(NamedTuple):
    vertex_points: Any
    vertex_index_to_neighbours_offset: Any
    vertex_index_to_neighbours_size: Any
    vertex_neighbours: Any

def create_fast_graph(graph: Graph, neighbour: GraphNeighbour) -> FastGraph:
    flat_neighbour_offset: list[int] = []
    flat_neighbour_size: list[int] = []
    flat_neighbour: list[int] = []

    current_flat_neighbour_offset = 0

    for neighbour_el in neighbour.neighbours:
        flat_neighbour_offset.append(current_flat_neighbour_offset)
        flat_neighbour_size.append(len(neighbour_el))
        flat_neighbour.extend(neighbour_el)
        current_flat_neighbour_offset += len(neighbour_el)

    flat_vertices: list[float] = []
    for vertex in graph.vertices:
        flat_vertices.append(vertex.x)
        flat_vertices.append(vertex.y)

    return FastGraph(
        vertex_points = np.array(flat_vertices, dtype=np.float32),
        vertex_index_to_neighbours_offset = np.array(flat_neighbour_offset, dtype=np.int32),
        vertex_index_to_neighbours_size = np.array(flat_neighbour_size, dtype=np.int32),
        vertex_neighbours = np.array(flat_neighbour, dtype=np.int32)
    )

def compute_sspd_frechet_preperation(sspd: SSPD_Transit_Node, graph: Graph, graph_neighbour: GraphNeighbour, closeness: float, epsilon: float, boundary_range: Interval) -> tuple[SSPD_Transit_Node, list[TransitPairData], Any, Any, Any, Any]:
    """Compute the datastructure to efficiently query 1+epsilon approximation of graph and segment on a SSPD tree."""

    transit_pair_list: list[TransitPairData] = []
    transit_pair_list2: list[TransitPairData2] = []
    exponent_structure_index_to_sparse_index_list: list[list[int]] = []
    sparse_index_to_point_list: list[list[Point]] = []

    current_offset_exponent_structure_index_to_sparse_index_vector = 0
    current_offset_sparse_index_to_point_vector = 0
    current_offset_sparse_index_matrix = 0

    def compute_sspd_frechet_recurse(node: SSPD_Transit_Node) -> SSPD_Transit_Node:
        nonlocal current_offset_exponent_structure_index_to_sparse_index_vector, current_offset_sparse_index_to_point_vector, current_offset_sparse_index_matrix, transit_pair_list

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
                    transit_point = transit_pair.transit_point
                    sspd_point = transit_pair.sspd_point

                    spine_distance_interval, _ = optimize_frechet_distance_graph_segment(graph.vertices[transit_pair.sspd_point], graph.vertices[transit_pair.transit_point], transit_pair.sspd_point, transit_pair.transit_point, graph, graph_neighbour, closeness, boundary_range)
                    spine_distance = interval_mid(spine_distance_interval)
                    assert spine_distance is not None

                    if spine_distance > 0:
                        exp_alpha = epsilon * spine_distance * 0.5
                        exp_beta = spine_distance / epsilon

                        exp_level_count = math.ceil(math.log2(exp_beta / exp_alpha))

                        # (lambda/2) / ((epsilon * lambda)/(4sqrt(2)))
                        exp_grid_radius_count = math.ceil(2 * math.sqrt(2) / epsilon)
                        exp_grid_side_count = 2 * exp_grid_radius_count + 1
    
                        exponent_structure_index_to_sparse_index: list[int] = []
                        sparse_index_to_point: list[Point] = []

                        for i in range(exp_level_count):
                            for dy_i in range(0, exp_grid_side_count):
                                for dx_i in range(0, exp_grid_side_count):
                                    # If i = 0, then fill everything.
                                    # If i > 0, then ignore points with index i-1 or less.
                                    dx = dx_i - exp_grid_radius_count
                                    dy = dy_i - exp_grid_radius_count
                                    x = dx * (2 ** (2 + i)) * exp_alpha * (epsilon / (4 * math.sqrt(2)))
                                    y = dy * (2 ** (2 + i)) * exp_alpha * (epsilon / (4 * math.sqrt(2)))
                                    r = math.sqrt(x ** 2 + y ** 2)

                                    if i > 0:
                                        if not epsilon_equal(r, 0):
                                            index = dx_i + dy_i * exp_grid_radius_count + i * exp_grid_radius_count * exp_grid_radius_count
                                            ixy = max(0, math.floor(math.log2(r / exp_alpha)))

                                            if ixy == index:
                                                exponent_structure_index_to_sparse_index.append(len(sparse_index_to_point))
                                                sparse_index_to_point.append(Point(x, y))
                                            else:
                                                exponent_structure_index_to_sparse_index.append(-1)
                                        else:
                                            exponent_structure_index_to_sparse_index.append(-1)
                                    else:
                                        exponent_structure_index_to_sparse_index.append(len(sparse_index_to_point))
                                        sparse_index_to_point.append(Point(x, y))

                        exponent_structure_index_to_sparse_index_list.append(exponent_structure_index_to_sparse_index)
                        sparse_index_to_point_list.append(sparse_index_to_point)
                        transit_pair_index = len(transit_pair_list)
                        new_transit_pair.append(TransitPair(transit_pair.transit_point, transit_pair.sspd_point, transit_pair_index))

                        transit_pair_list.append(TransitPairData(
                            sspd_point=sspd_point,
                            transit_point=transit_point,
                            exp_level_count=exp_level_count,
                            exp_grid_side_count=exp_grid_side_count,
                            exp_grid_radius_count=exp_grid_radius_count,
                            matrix_dimension=len(sparse_index_to_point),
                            offset_exponent_structure_index_to_sparse_index_vector=current_offset_exponent_structure_index_to_sparse_index_vector,
                            offset_sparse_index_to_point_vector=current_offset_sparse_index_to_point_vector,
                            offset_sparse_index_matrix=current_offset_sparse_index_matrix,
                            size_exponent_structure_index_to_sparse_index_vector=len(exponent_structure_index_to_sparse_index),
                            size_sparse_index_to_point_vector=len(sparse_index_to_point),
                            size_sparse_index_matrix=len(sparse_index_to_point) * len(sparse_index_to_point)
                        ))

                        transit_pair_list2.append(TransitPairData2(
                            exp_alpha=exp_alpha,
                            exp_beta=exp_beta
                        ))

                        current_offset_exponent_structure_index_to_sparse_index_vector += len(exponent_structure_index_to_sparse_index)
                        current_offset_sparse_index_to_point_vector += len(sparse_index_to_point)
                        current_offset_sparse_index_matrix += len(sparse_index_to_point) * len(sparse_index_to_point)
                    else:
                        exponent_structure_index_to_sparse_index_list.append([])
                        sparse_index_to_point_list.append([])
                        transit_pair_index = len(transit_pair_list)
                        new_transit_pair.append(TransitPair(transit_pair.transit_point, transit_pair.sspd_point, transit_pair_index))

                        transit_pair_list.append(TransitPairData(
                            sspd_point=sspd_point,
                            transit_point=transit_point,
                            exp_level_count=0,
                            exp_grid_side_count=0,
                            exp_grid_radius_count=0,
                            matrix_dimension=0,
                            offset_exponent_structure_index_to_sparse_index_vector=current_offset_exponent_structure_index_to_sparse_index_vector,
                            offset_sparse_index_to_point_vector=current_offset_sparse_index_to_point_vector,
                            offset_sparse_index_matrix=current_offset_sparse_index_matrix,
                            size_exponent_structure_index_to_sparse_index_vector=0,
                            size_sparse_index_to_point_vector=0,
                            size_sparse_index_matrix=0
                        ))

                        transit_pair_list2.append(TransitPairData2(
                            exp_alpha=0,
                            exp_beta=0
                        ))

                return SSPD_Transit_LeafNode(
                    node.weight_class,
                    node.region,
                    node.point_set,
                    node.transit_points,
                    new_transit_pair
                )

    new_sspd = compute_sspd_frechet_recurse(sspd)
    exponent_structure_index_to_sparse_index_vector = np.zeros(current_offset_exponent_structure_index_to_sparse_index_vector, dtype=np.int32)
    sparse_index_to_point_vector = np.zeros(current_offset_sparse_index_to_point_vector * 2, dtype=np.float32)
    sparse_index_matrix = np.zeros(current_offset_sparse_index_matrix, dtype=np.float32)

    for i in range(len(transit_pair_list)):
        transit_pair = transit_pair_list[i]
        exponent_structure_index_to_sparse_index = exponent_structure_index_to_sparse_index_list[i]
        sparse_index_to_point = sparse_index_to_point_list[i]

        for j, sparse_index in enumerate(exponent_structure_index_to_sparse_index):
            if sparse_index == -1:
                exponent_structure_index_to_sparse_index_vector[transit_pair.offset_exponent_structure_index_to_sparse_index_vector + j] = -1
            else:
                exponent_structure_index_to_sparse_index_vector[transit_pair.offset_exponent_structure_index_to_sparse_index_vector + j] = sparse_index

        for j, point in enumerate(sparse_index_to_point):
            sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + j) * 2 + 0] = point.x
            sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + j) * 2 + 1] = point.y

    return (new_sspd, transit_pair_list, transit_pair_list2, exponent_structure_index_to_sparse_index_vector, sparse_index_to_point_vector, sparse_index_matrix)

def compute_sspd_frechet_fill(graph: Graph, graph_neighbour: GraphNeighbour, transit_pair_list: list[TransitPairData], sparse_index_to_point_vector: Any, sparse_index_matrix: Any, closeness: float, boundary_range: Interval):
    progress_bar = tqdm(total=sparse_index_matrix.size)

    for transit_pair in transit_pair_list:
        for i in range(transit_pair.size_sparse_index_to_point_vector):
            for j in range(transit_pair.size_sparse_index_to_point_vector):
                pi = Point(sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + i) * 2 + 0], sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + i) * 2 + 1])
                pj = Point(sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + j) * 2 + 0], sparse_index_to_point_vector[(transit_pair.offset_sparse_index_to_point_vector + j) * 2 + 1])

                interval, _ = optimize_frechet_distance_graph_segment(pi, pj, transit_pair.transit_point, transit_pair.sspd_point, graph, graph_neighbour, closeness, boundary_range)
                distance_pp_qq = interval_mid(interval)

                if distance_pp_qq != -1:
                    sparse_index_matrix[transit_pair.offset_sparse_index_matrix + i * transit_pair.size_sparse_index_to_point_vector] = distance_pp_qq
                else:
                    sparse_index_matrix[transit_pair.offset_sparse_index_matrix + i * transit_pair.size_sparse_index_to_point_vector] = INFINITY

                progress_bar.update(1)

def compute_sspd_frechet_revert(
        sspd: SSPD_Transit_Node,
        graph: Graph,
        graph_neighbour: GraphNeighbour,
        exponent_structure_index_to_sparse_index_vector: Any,
        sparse_index_to_point_vector: Any,
        sparse_index_matrix: Any,
        transit_pair_lists: list[TransitPairData],
        transit_pair_lists2: list[TransitPairData2],
        epsilon: float,
        closeness: float,
        boundary_range: Interval
) -> SSPD_Transit_Node:
    """Compute the datastructure to efficiently query 1+epsilon approximation of graph and segment on a SSPD tree."""

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
                    spine_distance_interval, _ = optimize_frechet_distance_graph_segment(graph.vertices[transit_pair.sspd_point], graph.vertices[transit_pair.transit_point], transit_pair.sspd_point, transit_pair.transit_point, graph, graph_neighbour, closeness, boundary_range)
                    spine_distance = interval_mid(spine_distance_interval)
                    assert spine_distance is not None
                    ind: int = transit_pair.data
                    transit_pair_list = transit_pair_lists[ind]
                    transit_pair_list2 = transit_pair_lists2[ind]

                    if epsilon_equal(spine_distance, 0):
                        data = Epsilon_Frechet_Exact(spine_distance)
                    else:
                        exponent_structure_index_to_index: list[int | None] = []
                        index_to_point: list[Point] = []
                        exponent_distance_matrix: list[float | None] = []

                        for i in range(transit_pair_list.offset_exponent_structure_index_to_sparse_index_vector, transit_pair_list.offset_exponent_structure_index_to_sparse_index_vector+transit_pair_list.size_exponent_structure_index_to_sparse_index_vector):
                            el = exponent_structure_index_to_sparse_index_vector[i]
                            if el != -1:
                                exponent_structure_index_to_index.append(el)
                            else:
                                exponent_structure_index_to_index.append(None)

                        for i in range(transit_pair_list.offset_sparse_index_to_point_vector, transit_pair_list.offset_sparse_index_to_point_vector+transit_pair_list.size_sparse_index_to_point_vector):
                            el = Point(sparse_index_to_point_vector[i * 2 + 0], sparse_index_to_point_vector[i * 2 + 1])
                            index_to_point.append(el)

                        for i in range(transit_pair_list.offset_sparse_index_matrix, transit_pair_list.offset_sparse_index_matrix+transit_pair_list.size_sparse_index_matrix):
                            el = sparse_index_matrix[i]
                            if el < INFINITY:
                                exponent_distance_matrix.append(el)
                            else:
                                exponent_distance_matrix.append(None)

                        data = Epsilon_Frechet_Structure(
                            spine_distance,
                            epsilon=epsilon,
                            exponent_structure_index_to_index=exponent_structure_index_to_index,
                            exponent_structure_interval=Interval(transit_pair_list2.exp_alpha, transit_pair_list2.exp_beta),
                            exponent_structure_size=(
                                transit_pair_list.exp_grid_radius_count,
                                transit_pair_list.exp_grid_side_count,
                                transit_pair_list.exp_level_count),
                            index_to_point=index_to_point,
                            exponent_distance_matrix=exponent_distance_matrix,
                            exponent_distance_matrix_row_and_column_size=transit_pair_list.matrix_dimension,
                        )

                    new_transit_pair.append(TransitPair(transit_pair.transit_point, transit_pair.sspd_point, data))

                return SSPD_Transit_LeafNode(
                    node.weight_class,
                    node.region,
                    node.point_set,
                    node.transit_points,
                    new_transit_pair
                )

    return compute_sspd_frechet_recurse(sspd)
