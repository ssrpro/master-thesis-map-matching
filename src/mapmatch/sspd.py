"""Transit points are special vertices that must pass through a semi separated pair build using a BAR tree. These are useful for calculating
pairs of Frechet distance with efficient space storage."""

from __future__ import annotations

import math
from typing import Any, cast
from dataclasses import dataclass

import networkx as nx

from .bar_tree import Point, PointSet, Region, RegionCut, BarTreeNode, BarTreeOneCutNode, BarTreeTwoCutNode, BarTreeLeafNode
from .utilities import epsilon_geq, integer_range
from .input_graph import Graph, GraphNeighbour

@dataclass(frozen=True)
class ReducedPointSet:
    """A structure of lists containing a subset of the input point set using indices."""
    original_point_list: list[Point]
    permutation: list[int]

    def get_point_list(self) -> list[Point]:
        """Returns list of points"""
        return [self.original_point_list[ind] for ind in self.permutation]

    @property
    def n(self) -> int:
        """Returns the number of points the subset point set contains."""
        return len(self.permutation)

    @property
    def N(self) -> int:
        """Returns the number of points the original point set contains."""
        return len(self.original_point_list)

    @property
    def weight_class(self) -> int:
        """Weight class of a point set, which needs to equal the depth of the tree."""
        return int(math.floor(math.log2(self.N / self.n)))

    @staticmethod
    def create_from_point_set(point_set: PointSet) -> ReducedPointSet:
        """Create reduced point set from BAR tree point set."""
        return ReducedPointSet(point_set.original_point_list, point_set.permutation_sorted_x)

    @staticmethod
    def get_weight_class_from_point_set(point_set: PointSet) -> int:
        """Weight class of a point set, which needs to equal the depth of the tree."""
        return int(math.floor(math.log2(point_set.N / point_set.n)))

@dataclass(frozen=True)
class GhostInternalNode:
    """Internal node of the recursive data structure."""
    region: Region
    cut: RegionCut
    children: tuple[GhostNode, GhostNode]

@dataclass(frozen=True)
class GhostLeafNode:
    """Leaf node of the recursive data structure."""
    region: Region
    leaf_value: int

GhostNode = GhostInternalNode | GhostLeafNode
"""Efficient query datastructure to a child region."""

@dataclass(frozen=True)
class ReducedBarTreeInternalNode:
    """Describe a reduced BAR tree node."""
    weight_class: int
    region: Region
    point_set: ReducedPointSet
    recursive_structure: GhostNode
    children: list[ReducedBarTreeNode]

@dataclass(frozen=True)
class ReducedBarTreeLeafNode:
    """Describe a reduced BAR tree node."""
    weight_class: int
    region: Region
    point_set: ReducedPointSet

ReducedBarTreeNode = ReducedBarTreeLeafNode | ReducedBarTreeInternalNode

def get_ghost_node(parent_weight: int, node: BarTreeNode, children_list: list[tuple[int, BarTreeNode]]) -> GhostNode:
    """Constructs a recursive structure until it reaches a weight class that is strictly less than the input parent weight."""

    node_weight = ReducedPointSet.get_weight_class_from_point_set(node.point_set)
    assert parent_weight <= node_weight

    if parent_weight == node_weight:
        assert isinstance(node, (BarTreeOneCutNode, BarTreeTwoCutNode))
        child_1 = get_ghost_node(parent_weight, node.children[0], children_list)
        child_2 = get_ghost_node(parent_weight, node.children[1], children_list)
        return GhostInternalNode(node.region, node.cut, (child_1, child_2))
    else: # parent_weight < node_weight
        bartree_child_index = len(children_list)
        children_list.append((node_weight, node))
        return GhostLeafNode(node.region, bartree_child_index)

def extend_node(region: Region, point_set: ReducedPointSet, parent_weight: int, child_weight: int, new_children_list: list[ReducedBarTreeNode]):
    """
    When the target weight class more than one larger than the parent weight class, create duplicate nodes one smaller weight class until it reaches the target.
    Visually, it looks like 2a -> 6b => 2a -> 3b -> 4b -> 5b -> 6b.
    Returns the children list of the last node that got extended which is the children list of 6b in the above example.
    """
    def get_ghost_singleton(region: Region) -> GhostNode:
        return GhostLeafNode(region, 0)

    new_children_list_target: list[ReducedBarTreeNode] = new_children_list

    # Extend weight in [w_p+1, w_c)
    # 0P <- 3C  ==>  0P <- 1E <- 2E <- 3C
    for weight in range(parent_weight+1, child_weight):
        new_children_list_next_target: list[ReducedBarTreeNode] = []
        node_target = ReducedBarTreeInternalNode(weight, region, point_set, get_ghost_singleton(region), new_children_list_next_target)
        new_children_list_target.append(node_target)
        new_children_list_target = new_children_list_next_target

    return new_children_list_target

def compute_modified_bar_tree(node: BarTreeNode) -> ReducedBarTreeNode:
    """Computes recursively a modified BAR tree that guarantees weight class equal to the depth of the tree."""
    region = node.region.get_reduced_region()
    point_set = ReducedPointSet.create_from_point_set(node.point_set)
    node_weight = point_set.weight_class

    match node:
        case BarTreeOneCutNode() | BarTreeTwoCutNode():
            children_list: list[tuple[int, BarTreeNode]] = []
            new_children_list: list[ReducedBarTreeNode] = []
            recursive_structure = get_ghost_node(node_weight, node, children_list)

            for child_weight, child in children_list:
                child_region = child.region.get_reduced_region()
                child_point_set = ReducedPointSet.create_from_point_set(child.point_set)

                child_new_children_list = extend_node(child_region, child_point_set, node_weight, child_weight, new_children_list)
                child_new_children_list.append(compute_modified_bar_tree(child))

            return ReducedBarTreeInternalNode(node_weight, region, point_set, recursive_structure, new_children_list)
        case BarTreeLeafNode():
            return ReducedBarTreeLeafNode(node_weight, region, point_set)

@dataclass(frozen=True)
class SSPD_Transit_InternalNode:
    """Describes a semi separated decomposition internal node that includes a matrix to go through the children."""
    weight_class: int
    region: tuple[Region, Region]
    recursive_structure: tuple[GhostNode, GhostNode]
    matrix_NM: tuple[int, int]
    matrix: list[SSPD_Transit_Node]

@dataclass(frozen=True)
class SSPD_Transit_LeafNode:
    """Describes a semi separated decomposition leaf node that includes semi separated pair and their transit pairs."""
    weight_class: int
    region: tuple[Region, Region]
    point_set: tuple[ReducedPointSet, ReducedPointSet]
    transit_points: list[int]
    transit_pair: list[TransitPair]

@dataclass(frozen=True)
class TransitPair:
    """Transit pair data structure with extra variable to store additional structure."""
    transit_point: int
    sspd_point: int
    data: Any

@dataclass(frozen=True)
class SSPD_Info:
    """Contains number of sspds and number of transit pairs."""
    sspd_count: int
    transit_pair_count: int

SSPD_Transit_Node = SSPD_Transit_InternalNode | SSPD_Transit_LeafNode

def compute_sspd(graph: Graph, graph_neighbour: GraphNeighbour, node: ReducedBarTreeNode, s: float) -> SSPD_Transit_Node:
    """Compute SSPD from a reduced BAR tree with a separating constant."""
    match node:
        case ReducedBarTreeInternalNode():
            ss_matrix_N = len(node.children)
            ss_matrix_mod: list[SSPD_Transit_Node | None] = [None] * (ss_matrix_N * ss_matrix_N)

            for i in range(ss_matrix_N):
                for j in range(ss_matrix_N):
                    if i != j:
                        ss_matrix_mod[i * ss_matrix_N + j] = find_ss_pairs(graph, graph_neighbour, node.children[i], node.children[j], s, 1)
                    else: # i == j:
                        ss_matrix_mod[i * ss_matrix_N + j] = compute_sspd(graph, graph_neighbour, node.children[i], s)

            ss_matrix = cast(list[SSPD_Transit_Node], ss_matrix_mod)
            return SSPD_Transit_InternalNode(node.weight_class, (node.region, node.region), (node.recursive_structure, node.recursive_structure), (ss_matrix_N, ss_matrix_N), ss_matrix)
        case ReducedBarTreeLeafNode():
            transit_points, transit_pairs = compute_transit_nx(graph, node.point_set, node.point_set)
            return SSPD_Transit_LeafNode(node.weight_class, (node.region, node.region), (node.point_set, node.point_set), transit_points, transit_pairs)

def find_ss_pairs(graph: Graph, graph_neighbour: GraphNeighbour, node1: ReducedBarTreeNode, node2: ReducedBarTreeNode, s: float, depth: int) -> SSPD_Transit_Node:
    """Find semi separated pairs from two nodes of a reduced BAR tree."""
    assert node1.weight_class == node2.weight_class, "Weight class should be equal."
    weight = node1.weight_class

    bounding_region1 = Region.create_region_from_point_list(node1.point_set.get_point_list())
    bounding_region2 = Region.create_region_from_point_list(node2.point_set.get_point_list())

    if epsilon_geq(Region.get_distance(bounding_region1, bounding_region2), s * min(bounding_region1.get_diam() / 2, bounding_region2.get_diam() / 2)):
        transit_points, transit_pairs = compute_transit_nx(graph, node1.point_set, node2.point_set)
        return SSPD_Transit_LeafNode(weight, (node1.region, node2.region), (node1.point_set, node2.point_set), transit_points, transit_pairs)
    if isinstance(node1, ReducedBarTreeLeafNode) or isinstance(node2, ReducedBarTreeLeafNode):
        assert isinstance(node1, ReducedBarTreeLeafNode) and isinstance(node2, ReducedBarTreeLeafNode), "If one node is a leaf, both needs to be leaves at the same time."
        transit_points, transit_pairs = compute_transit_nx(graph, node1.point_set, node2.point_set)
        return SSPD_Transit_LeafNode(weight, (node1.region, node2.region), (node1.point_set, node2.point_set), transit_points, transit_pairs)
    else:
        ss_matrix_N = len(node1.children)
        ss_matrix_M = len(node2.children)
        ss_matrix_mod: list[SSPD_Transit_Node | None] = [None] * (ss_matrix_N * ss_matrix_M)

        for i in range(ss_matrix_N):
            for j in range(ss_matrix_M):
                ss_matrix_mod[i * ss_matrix_M + j] = find_ss_pairs(graph, graph_neighbour, node1.children[i], node2.children[j], s, depth+1)

        ss_matrix = cast(list[SSPD_Transit_Node], ss_matrix_mod)
        return SSPD_Transit_InternalNode(weight, (node1.region, node2.region), (node1.recursive_structure, node2.recursive_structure), (ss_matrix_N, ss_matrix_M), ss_matrix)

def compute_transit_nx(graph: Graph, point_set_1: ReducedPointSet, point_set_2: ReducedPointSet) -> tuple[list[int], list[TransitPair]]:
    G = nx.Graph()
    illegal_points: set[int] = set()

    for i, j in graph.edges:
        G.add_edge(i, j, capacity=1) # type: ignore
    for i in point_set_1.permutation:
        G.add_edge(-1, i, capacity=len(graph.vertices)) # type: ignore
        illegal_points.add(i)
    for i in point_set_2.permutation:
        G.add_edge(-2, i, capacity=len(graph.vertices)) # type: ignore
        illegal_points.add(i)

    _, partition = nx.minimum_cut(G, -1, -2) # type: ignore
    reachable, non_reachable = partition # type: ignore
    transit_edges: set[tuple[int, int]] = set() # type: ignore
    for u, nbrs in ((n, G[n]) for n in reachable): # type: ignore
        transit_edges.update((u, v) for v in nbrs if v in non_reachable) # type: ignore

    transit_points: list[int] = []
    transit_pairs: list[TransitPair] = []

    for i, j in transit_edges:
        if i < 0 or j < 0:
            continue
        if i not in illegal_points:
            transit_points.append(i)
        elif j not in illegal_points:
            transit_points.append(j)
        else:
            transit_pairs.append(TransitPair(i, j, None)) # <-- Degenerate case

    for transit_point in transit_points:
        for u in point_set_1.permutation:
            transit_pairs.append(TransitPair(transit_point, u, None))
        for u in point_set_2.permutation:
            transit_pairs.append(TransitPair(transit_point, u, None))

    return (transit_points, transit_pairs)

def compute_transit(graph: Graph, graph_neighbour: GraphNeighbour, point_set_1: ReducedPointSet, point_set_2: ReducedPointSet) -> tuple[list[int], list[TransitPair]]:
    """Calculates the transit vertices and transit pairs for a semi separated pair."""

    capacity: dict[int, int] = dict()
    flow: dict[int, int] = dict()
    V = len(graph.vertices) + 2
    vi_source = V - 2
    vi_sink = V - 1

    source: set[int] = set()
    sink: set[int] = set()
    path_structure: list[int] = []

    for i in range(V):
        path_structure.append(0)

    for i, j in graph.edges:
        # Assume (i, j) in E ==> (j, i) in E
        capacity[i * V + j] = 1
        flow[i * V + j] = 0
    for i in point_set_1.permutation:
        capacity[vi_source * V + i] = V
        flow[vi_source * V + i] = 0
        flow[i * V + vi_source] = 0
        source.add(i)
    for i in point_set_2.permutation:
        capacity[i * V + vi_sink] = V
        flow[vi_sink * V + i] = 0
        flow[i * V + vi_sink] = 0
        sink.add(i)

    def get_residue(u: int, v: int) -> int:
        current_capacity = 0
        if u * V + v in capacity:
            current_capacity = capacity[u * V + v]

        current_flow = flow[u * V + v]
        current_residue = current_capacity - current_flow
        return current_residue

    def find_path() -> int | None:
        queue: list[int] = []
        visited: set[int] = set()

        for source_index in source:
            queue.append(source_index)
            visited.add(source_index)

        while len(queue) > 0:
            vertex_index = queue.pop()
            
            if vertex_index in sink:
                return vertex_index
            else:
                for neighbour_index in graph_neighbour.get_vertex_neighbours(vertex_index):
                    if neighbour_index not in visited:
                        if get_residue(vertex_index, neighbour_index) > 0:
                            visited.add(neighbour_index)
                            queue.append(neighbour_index)
                            path_structure[neighbour_index] = vertex_index

        return None

    def find_cut() -> set[int]:
        queue: list[int] = []
        visited: set[int] = set()
        cuts: set[int] = set()

        for source_index in source:
            queue.append(source_index)
            visited.add(source_index)

        while len(queue) > 0:
            vertex_index = queue.pop()
            
            assert vertex_index not in sink, "After searching on the residue graph, the sink by construction should not be reachable."

            for neighbour_index in graph_neighbour.get_vertex_neighbours(vertex_index):
                if neighbour_index not in visited:
                    if get_residue(vertex_index, neighbour_index) > 0:
                        visited.add(neighbour_index)
                        queue.append(neighbour_index)

        for u, v in graph.edges:
            if u in visited and v not in visited:
                cuts.add(u)

        return cuts

    ii = 0
    while (current_sink := find_path()) is not None:
        ii += 1

        delta = V

        pointer = current_sink
        while pointer not in source:
            prev = path_structure[pointer]
            delta = min(delta, get_residue(prev, pointer))
            pointer = prev

        pointer = current_sink
        while pointer not in source:
            prev = path_structure[pointer]
            flow[prev * V + pointer] += delta
            flow[pointer * V + prev] -= delta
            pointer = prev

    transit_points = list(find_cut())
    transit_pairs: list[TransitPair] = []

    for transit_point in transit_points:
        for u in point_set_1.permutation:
            transit_pairs.append(TransitPair(transit_point, u, None))
        for u in point_set_2.permutation:
            transit_pairs.append(TransitPair(transit_point, u, None))

    return (transit_points, transit_pairs)

def get_sspd_info(sspd: SSPD_Transit_Node) -> SSPD_Info:
    sspd_count = 0
    transit_pair_count = 0

    def recurse_sspd_info(node: SSPD_Transit_Node):
        nonlocal sspd_count, transit_pair_count

        match node:
            case SSPD_Transit_InternalNode():
                for child in node.matrix:
                    recurse_sspd_info(child)
            case SSPD_Transit_LeafNode():
                sspd_count += 1
                transit_pair_count += len(node.transit_pair)

    recurse_sspd_info(sspd)

    return SSPD_Info(
        sspd_count=sspd_count,
        transit_pair_count=transit_pair_count
    )

def query_recursive_structure(node: GhostNode, p: Point) -> int:
    """Query the correct child of a SSPD from a recursive structure."""
    assert node.region.point_inside(p)

    match node:
        case GhostLeafNode():
            return node.leaf_value
        case GhostInternalNode():
            choose_left = False

            match node.cut.axis:
                case "x":
                    choose_left = p.x < node.cut.value
                case "y":
                    choose_left = p.y < node.cut.value
                case "z":
                    choose_left = p.z < node.cut.value

            if choose_left:
                return query_recursive_structure(node.children[0], p)
            else:
                return query_recursive_structure(node.children[1], p)

def query_sspd(node: SSPD_Transit_Node, p1: Point, p2: Point) -> SSPD_Transit_LeafNode:
    """Query region and point set (A, B) such that p1 in A, p2 in B and (A, B) are semi separated."""
    assert node.region[0].point_inside(p1)
    assert node.region[1].point_inside(p2)

    match node:
        case SSPD_Transit_InternalNode():
            i = query_recursive_structure(node.recursive_structure[0], p1)
            j = query_recursive_structure(node.recursive_structure[1], p2)

            assert integer_range(i, 0, node.matrix_NM[0] - 1), f"{i} should be less than {node.matrix_NM[0]}"
            assert integer_range(j, 0, node.matrix_NM[1] - 1), f"{j} should be less than {node.matrix_NM[1]}"

            return query_sspd(node.matrix[i * node.matrix_NM[1] + j], p1, p2)
        case SSPD_Transit_LeafNode():
            return node
