"""
Defines datastructures for efficiently query n-dimensional boxes of size p intersecting points or points
of size p intersecting boxes. On all those datastructures the storage and construction time is O(np log^n p)
and query time is O(n log^n p).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .utilities import Interval, in_interval, in_hyperinterval, IntRange, PointND, HyperInterval, EPSILON, INFINITY, epsilon_geq, epsilon_leq, interval_intersect

BOUNDARY_EPSILON = 10 * EPSILON
"""
All segment trees contain interior intervals (a, b) and closed intervals [a, a]
To make this possible in floating point number system we instead store them as
(a + eps, b - eps) and (a - eps, a + eps) respectively, where eps is the boundary
epsilon, which should be a multiple of epsilon.
"""

@dataclass(frozen=True)
class RangeTreeNDLeafNodeFinal:
    """Leaf node of the n-dimensional range tree, where the component c = n."""
    key: float
    index: int
    component: int
    internal: list[int]

@dataclass(frozen=True)
class RangeTreeNDLeafNode:
    """Leaf node of the n-dimensional range tree, where the component c < n."""
    key: float
    index: int
    component: int
    internal: RangeTreeNDNode | RangeTreeNDNodeFinal | None

@dataclass(frozen=True)
class RangeTreeNDInternalNodeFinal:
    """Internal node of the n-dimensional range tree, where the component c = n."""
    key: float
    index: int
    component: int
    internal: list[int]
    left: RangeTreeNDNodeFinal
    right: RangeTreeNDNodeFinal

@dataclass(frozen=True)
class RangeTreeNDInternalNode:
    """Internal node of the n-dimensional range tree, where the component c < n."""
    key: float
    index: int
    component: int
    internal: RangeTreeNDNode | RangeTreeNDNodeFinal | None
    left: RangeTreeNDNode
    right: RangeTreeNDNode

RangeTreeNDNode = RangeTreeNDInternalNode | RangeTreeNDLeafNode
RangeTreeNDNodeFinal = RangeTreeNDInternalNodeFinal | RangeTreeNDLeafNodeFinal

@dataclass(frozen=True)
class RangeTreeND:
    """
    Range tree datastructure storing points for efficient range queries.
    Input are collection of points in n dimension. And query object is a n dimensional 
    hyperinterval C, where all point indices in C gets returned.
    """
    root: RangeTreeNDNode
    points: list[PointND]
    dimension: int

def create_rangetree_nd(points: list[PointND], dimension: int) -> RangeTreeND:
    """Create an n-dimensional range tree datastructure from a list of p points in O(np log^n p) time."""

    def sort_indices(indices: list[int], points: list[PointND], component: int) -> list[int]:
        points_structure: list[tuple[float, int]] = []

        for index in indices:
            points_structure.append((points[index][component], index))

        points_structure = sorted(points_structure, key=lambda p: p[0])
        indices = [index for _, index in points_structure]
        return indices

    def create_rangetree_nd_recurse_final(ls: IntRange, indices: list[int], points: list[PointND], component: int) -> RangeTreeNDNodeFinal:
        range_size = ls.end - ls.begin

        assert range_size >= 1, "Invariant: All ranges must include at least one index."

        if range_size == 1:
            index = indices[ls.begin]
            key = points[index][component]
            return RangeTreeNDLeafNodeFinal(key, index, component, [index])
        else: # range_size > 1
            m = ls.begin + math.floor(range_size / 2)
            index = indices[m]
            key = points[index][component]

            left = create_rangetree_nd_recurse_final(IntRange(ls.begin, m), indices, points, component)
            right = create_rangetree_nd_recurse_final(IntRange(m, ls.end), indices, points, component)

            assert left is not None and right is not None, "Children must be non empty."

            internal_indices = indices[ls.begin:ls.end]
            internal_indices = sort_indices(internal_indices, points, component)
            internal = indices[ls.begin:ls.end]
            return RangeTreeNDInternalNodeFinal(key, index, component, internal, left, right)

    def create_rangetree_nd_recurse(ls: IntRange, indices: list[int], points: list[PointND], component: int) -> RangeTreeNDNode | None:
        range_size = ls.end - ls.begin

        assert range_size >= 1, "Invariant: All ranges must include at least one index."

        if range_size == 1:
            index = indices[ls.begin]
            key = points[index][component]

            internal: RangeTreeNDNode | RangeTreeNDNodeFinal | None = None
            if component+1 == dimension-1:
                internal = create_rangetree_nd_recurse_final(IntRange(0, 1), [index], points, component+1)
            else:
                internal = create_rangetree_nd_recurse(IntRange(0, 1), [index], points, component+1)

            return RangeTreeNDLeafNode(key, index, component, internal)
        else: # range_size > 1
            m = ls.begin + math.floor(range_size / 2)
            index = indices[m]
            key = points[index][component]

            left = create_rangetree_nd_recurse(IntRange(ls.begin, m), indices, points, component)
            right = create_rangetree_nd_recurse(IntRange(m, ls.end), indices, points, component)

            assert left is not None and right is not None, "Children must be non empty."

            internal_indices = indices[ls.begin:ls.end]
            internal_indices = sort_indices(internal_indices, points, component+1)
            internal: RangeTreeNDNode | RangeTreeNDNodeFinal | None = None
            if component+1 == dimension-1:
                internal = create_rangetree_nd_recurse_final(IntRange(0, len(internal_indices)), internal_indices, points, component+1)
            else:
                internal = create_rangetree_nd_recurse(IntRange(0, len(internal_indices)), internal_indices, points, component+1)

            return RangeTreeNDInternalNode(key, index, component, internal, left, right)

    indices = list(range(len(points)))
    indices = sort_indices(indices, points, 0)
    root = create_rangetree_nd_recurse(IntRange(0, len(indices)), indices, points, 0)
    assert len(points) > 0, "Input point list should not be empty."
    assert root is not None, "If point list is not empty, then the root tree should not be not None."
    return RangeTreeND(root, points, dimension)

def query_rangetree_nd(tree: RangeTreeND, bound: HyperInterval) -> list[int]:
    """Queries on an n-dimensional range tree datastructure from a list of p points in O(n log^n p) time."""

    dimension = tree.dimension
    points = tree.points

    def query_rangetree_nd_recurse(node: RangeTreeNDNode | RangeTreeNDNodeFinal | None) -> list[int]:
        if node is None:
            return []

        result: list[int] = []
        bound_interval = bound[node.component]

        split = node
        while isinstance(split, (RangeTreeNDInternalNode, RangeTreeNDInternalNodeFinal)) and not in_interval(split.key, bound_interval):
            if bound_interval.left <= split.key:
                assert bound_interval.left <= split.key and bound_interval.right <= split.key
                split = split.left
            else:
                assert split.key <= bound_interval.left and split.key <= bound_interval.right
                split = split.right

        if not isinstance(split, (RangeTreeNDInternalNode, RangeTreeNDInternalNodeFinal)):
            if in_hyperinterval(points[split.index], bound, dimension):
                return [split.index]
        else:
            if in_hyperinterval(points[split.index], bound, dimension):
                result.append(split.index)

            gray_left = split.left
            while isinstance(gray_left, (RangeTreeNDInternalNode, RangeTreeNDInternalNodeFinal)):
                if in_hyperinterval(points[gray_left.index], bound, dimension):
                    result.append(gray_left.index)

                match gray_left:
                    case RangeTreeNDInternalNode():
                        if bound_interval.left <= gray_left.key:
                            result.extend(query_rangetree_nd_recurse(gray_left.right.internal))
                            gray_left = gray_left.left
                        else:
                            gray_left = gray_left.right
                    case RangeTreeNDInternalNodeFinal():
                        if bound_interval.left <= gray_left.key:
                            result.extend(gray_left.right.internal)
                            gray_left = gray_left.left
                        else:
                            gray_left = gray_left.right

            if in_hyperinterval(points[gray_left.index], bound, dimension):
                result.append(gray_left.index)

            gray_right = split.right
            while not isinstance(gray_right, (RangeTreeNDLeafNode, RangeTreeNDLeafNodeFinal)):
                if in_hyperinterval(points[gray_right.index], bound, dimension):
                    result.append(gray_right.index)

                match gray_right:
                    case RangeTreeNDInternalNode():
                        if bound_interval.right >= gray_right.key:
                            result.extend(query_rangetree_nd_recurse(gray_right.left.internal))
                            gray_right = gray_right.right
                        else:
                            gray_right = gray_right.left
                    case RangeTreeNDInternalNodeFinal():
                        if bound_interval.right >= gray_right.key:
                            result.extend(gray_right.left.internal)
                            gray_right = gray_right.right
                        else:
                            gray_right = gray_right.left

            if in_hyperinterval(points[gray_right.index], bound, dimension):
                result.append(gray_right.index)

        return result

    return query_rangetree_nd_recurse(tree.root)

def query_rangetree_nd_naive(tree: RangeTreeND, bound: HyperInterval) -> list[int]:
    """Slow version of query on an n-dimensional range tree of p points in O(pn) time."""

    indices: list[int] = []

    for index, point in enumerate(tree.points):
        if in_hyperinterval(point, bound, tree.dimension):
            indices.append(index)

    return indices

def size_rangetree_nd(tree: RangeTreeND):
    """Returns the total number of point indices inside the range tree."""

    def size_rangetree_nd_recurse(node: RangeTreeNDNode | RangeTreeNDNodeFinal | None) -> int:
        match node:
            case RangeTreeNDInternalNode():
                return size_rangetree_nd_recurse(node.left) + size_rangetree_nd_recurse(node.right) + size_rangetree_nd_recurse(node.internal)
            case RangeTreeNDLeafNode():
                return size_rangetree_nd_recurse(node.internal)
            case RangeTreeNDInternalNodeFinal():
                return size_rangetree_nd_recurse(node.left) + size_rangetree_nd_recurse(node.right) + len(node.internal)
            case RangeTreeNDLeafNodeFinal():
                return len(node.internal)
            case None:
                return 0
            
    return size_rangetree_nd_recurse(tree.root)

@dataclass(frozen=True)
class SegmentTree1DInternalNode:
    """Internal node of a 1 dimensional segment tree."""
    interval: Interval
    internal: list[int]
    left: SegmentTree1DNode
    right: SegmentTree1DNode

@dataclass(frozen=True)
class SegmentTree1DLeaf:
    """Leaf node of a 1 dimensional segment tree."""
    interval: Interval
    internal: list[int]

SegmentTree1DNode = SegmentTree1DInternalNode | SegmentTree1DLeaf

@dataclass(frozen=True)
class SegmentTree1D:
    """Segment tree on a given list of intervals."""
    root: SegmentTree1DNode
    intervals: list[Interval]

def create_segmenttree_1d(intervals: list[Interval]) -> SegmentTree1D:
    """Create a 1 dimensional segment tree of p points in O(p log p) time."""

    assert len(intervals) > 0, "Segment tree must have at least one interval."

    # Create sorted interval endpoints to construct elementary intervals.
    interval_endpoints: list[float] = []
    for interval in intervals:
        interval_endpoints.append(interval.left)
        interval_endpoints.append(interval.right)
    interval_endpoints = sorted(interval_endpoints)

    # Construct elementary intervals in the form (-∞, a) [a, a] (a, b) [b, b] ... [z, z] (z, ∞).
    elementary_intervals: list[Interval] = []
    elementary_intervals.append(Interval(-INFINITY, interval_endpoints[0] - BOUNDARY_EPSILON))
    for i in range(len(interval_endpoints)):
        elementary_intervals.append(Interval(interval_endpoints[i] - BOUNDARY_EPSILON, interval_endpoints[i] + BOUNDARY_EPSILON))
        if i + 1 < len(interval_endpoints):
            elementary_intervals.append(Interval(interval_endpoints[i] + BOUNDARY_EPSILON, interval_endpoints[i+1] - BOUNDARY_EPSILON))
    elementary_intervals.append(Interval(interval_endpoints[-1], INFINITY))

    # Create binary tree bottom-up using elementary intervals as leaves.
    leaf_nodes: list[SegmentTree1DNode] = [SegmentTree1DLeaf(elementary_interval, []) for elementary_interval in elementary_intervals]
    remainder: SegmentTree1DNode | None = None
    nodes = leaf_nodes

    while len(nodes) > 1 or remainder is not None:
        assert len(nodes) != 0, "Technical invariant, degenerate case: len(nodes) == 0 and remainder != None should never occur."

        if remainder is not None:
            nodes.append(remainder)

        new_nodes: list[SegmentTree1DNode] = []
        for pair_i in range(0, len(nodes) - 1, 2):
            node_left = nodes[pair_i]
            node_right = nodes[pair_i+1]

            assert epsilon_geq(node_left.interval.right, node_right.interval.left), "Invariant that I, the union of two child intervals Ia, Ib is connected => I is an interval."
            new_node = SegmentTree1DInternalNode(Interval(node_left.interval.left, node_right.interval.right), [], node_left, node_right)
            new_nodes.append(new_node)

        # Last node that did not get paired up will be the remainder of the next iteration.
        remainder = nodes[-1] if len(nodes) % 2 != 0 else None
        nodes = new_nodes

    assert len(nodes) == 1, "We must be left with a single node that represents the root of the constructed binary tree."
    root = nodes[0]

    def insert_segment(node: SegmentTree1DNode, interval_index: int, interval: Interval) -> None:
        # If Int(node) is subset of interval
        if epsilon_geq(node.interval.left, interval.left) and epsilon_leq(node.interval.right, interval.right):
            # Stop recursion and insert
            node.internal.append(interval_index)
        else:
            # Continue on children of node
            if not isinstance(node, SegmentTree1DInternalNode):
                return

            assert interval_intersect(node.interval, interval) is not None, "Invariant: Int(node) and interval must intersect."

            if interval_intersect(node.left.interval, interval) is not None:
                insert_segment(node.left, interval_index, interval)
            if interval_intersect(node.right.interval, interval) is not None:
                insert_segment(node.right, interval_index, interval)

    for interval_index, interval in enumerate(intervals):
        insert_segment(root, interval_index, interval)

    return SegmentTree1D(root, intervals)

def query_segmenttree_1d(segment_tree: SegmentTree1D, value: float) -> list[int]:
    """Query on a 1 dimensional segment tree of p points in O(log p) time."""

    def query_segmenttree_1d_recurse(node: SegmentTree1DNode, value: float) -> list[int]:
        reported_interval_indices: list[int] = []

        reported_interval_indices.extend(node.internal)
        
        if type(node) is SegmentTree1DInternalNode:
            if in_interval(value, node.left.interval):
                reported_interval_indices.extend(query_segmenttree_1d_recurse(node.left, value))
            else:
                assert in_interval(value, node.right.interval)
                reported_interval_indices.extend(query_segmenttree_1d_recurse(node.right, value))
        
        return reported_interval_indices

    return query_segmenttree_1d_recurse(segment_tree.root, value)

def query_segmenttree_1d_naive(tree: SegmentTree1D, point: float) -> list[int]:
    """Slow version of query on a 1 dimensional segment of p points in O(p) time."""

    indices: list[int] = []

    for index, interval in enumerate(tree.intervals):
        if in_interval(point, interval):
            indices.append(index)

    return indices

@dataclass(frozen=True)
class SegmentTreeNDInternalNodeFinal:
    """Internal node of an n dimensional segment tree."""
    interval: Interval
    internal: list[int]
    component: int
    left: SegmentTreeNDNodeFinal
    right: SegmentTreeNDNodeFinal

@dataclass(frozen=True)
class SegmentTreeNDLeafFinal:
    """Leaf node of an n dimensional segment tree."""
    interval: Interval
    internal: list[int]
    component: int

SegmentTreeNDNodeFinal = SegmentTreeNDInternalNodeFinal | SegmentTreeNDLeafFinal

@dataclass(frozen=True)
class SegmentTreeNDInternalNode:
    """Internal node of an n dimensional segment tree."""
    interval: Interval
    internal: SegmentTreeNDNodeFinal | SegmentTreeNDNode | None
    component: int
    left: SegmentTreeNDNode
    right: SegmentTreeNDNode

@dataclass(frozen=True)
class SegmentTreeNDLeaf:
    """Leaf node of an n dimensional segment tree."""
    interval: Interval
    internal: SegmentTreeNDNodeFinal | SegmentTreeNDNode | None
    component: int

SegmentTreeNDNode = SegmentTreeNDInternalNode | SegmentTreeNDLeaf

@dataclass(frozen=True)
class SegmentTreeND:
    """Segment tree on a given list of n dimensional intervals."""
    root: SegmentTreeNDNode
    hyper_intervals: list[HyperInterval]
    dimension: int

def create_segmenttree_nd(hyper_intervals: list[HyperInterval], dimension: int) -> SegmentTreeND:
    """Create an n dimensional segment tree of p points in O(np log^n p) time."""

    def transform_segmenttree_nd_recurse_final(node: SegmentTree1DNode, subset_indices: list[int], component: int) -> SegmentTreeNDNodeFinal:
        internal = [subset_indices[index] for index in node.internal]

        match node:
            case SegmentTree1DInternalNode():
                left = transform_segmenttree_nd_recurse_final(node.left, subset_indices, component)
                right = transform_segmenttree_nd_recurse_final(node.right, subset_indices, component)
                return SegmentTreeNDInternalNodeFinal(node.interval, internal, component, left, right)
            case SegmentTree1DLeaf():
                return SegmentTreeNDLeafFinal(node.interval, internal, component)

    def create_segmenttree_nd_recurse_final(subset_indices: list[int], component: int) -> SegmentTreeNDNodeFinal | None:
        if len(subset_indices) == 0:
            return None

        intervals = [hyper_intervals[subset_index][component] for subset_index in subset_indices]
        segment_tree_1d = create_segmenttree_1d(intervals)
        assert segment_tree_1d is not None and segment_tree_1d.root is not None, "Since input intervals list is non empty, the resulting segment_tree_1d should be not zero."

        return transform_segmenttree_nd_recurse_final(segment_tree_1d.root, subset_indices, component)

    def transform_segmenttree_nd_recurse(node: SegmentTree1DNode, subset_indices: list[int], component: int) -> SegmentTreeNDNode:
        subset_indices_internal = [subset_indices[index] for index in node.internal]
        internal: SegmentTreeNDNode | SegmentTreeNDNodeFinal | None = None

        if component == dimension - 2:
            internal = create_segmenttree_nd_recurse_final(subset_indices_internal, component+1)
        else:
            internal = create_segmenttree_nd_recurse(subset_indices_internal, component+1)

        match node:
            case SegmentTree1DInternalNode():
                left = transform_segmenttree_nd_recurse(node.left, subset_indices, component)
                right = transform_segmenttree_nd_recurse(node.right, subset_indices, component)
                return SegmentTreeNDInternalNode(node.interval, internal, component, left, right)
            case SegmentTree1DLeaf():
                return SegmentTreeNDLeaf(node.interval, internal, component)

    def create_segmenttree_nd_recurse(subset_indices: list[int], component: int) -> SegmentTreeNDNode | None:
        if len(subset_indices) == 0:
            return None

        intervals = [hyper_intervals[subset_index][component] for subset_index in subset_indices]
        segment_tree_1d = create_segmenttree_1d(intervals)
        assert segment_tree_1d is not None and segment_tree_1d.root is not None, "Since input intervals list is non empty, the resulting segment_tree_1d should be not zero."

        return transform_segmenttree_nd_recurse(segment_tree_1d.root, subset_indices, component)

    root = create_segmenttree_nd_recurse(list(range(len(hyper_intervals))), 0)
    assert len(hyper_intervals) > 0, "Input hyper intervals should not be empty."
    assert root is not None, "If hyper-intervals list is not empty, then the root tree should not be not None."
    return SegmentTreeND(root, hyper_intervals, dimension)

def query_segmenttree_nd(tree: SegmentTreeND, point: PointND) -> list[int]:
    """Queries on an n dimensional segment tree of p points in O(n log^n p) time."""

    def query_segmenttree_nd_recurse(node: SegmentTreeNDNode | SegmentTreeNDNodeFinal | None) -> list[int]:
        if node is None:
            return []

        reported_interval_indices: list[int] = []

        component = node.component
        value = point[component]

        match node:
            case SegmentTreeNDInternalNode():
                reported_interval_indices.extend(query_segmenttree_nd_recurse(node.internal))

                if in_interval(value, node.left.interval):
                    reported_interval_indices.extend(query_segmenttree_nd_recurse(node.left))
                else:
                    assert in_interval(value, node.right.interval)
                    reported_interval_indices.extend(query_segmenttree_nd_recurse(node.right))

            case SegmentTreeNDLeaf():
                reported_interval_indices.extend(query_segmenttree_nd_recurse(node.internal))

            case SegmentTreeNDInternalNodeFinal():
                reported_interval_indices.extend(node.internal)

                if in_interval(value, node.left.interval):
                    reported_interval_indices.extend(query_segmenttree_nd_recurse(node.left))
                else:
                    assert in_interval(value, node.right.interval)
                    reported_interval_indices.extend(query_segmenttree_nd_recurse(node.right))

            case SegmentTreeNDLeafFinal():
                reported_interval_indices.extend(node.internal)

        return reported_interval_indices

    return query_segmenttree_nd_recurse(tree.root)

def query_segmenttree_nd_naive(tree: SegmentTreeND, point: PointND) -> list[int]:
    """Slow version of query on a n dimensional segment of p points in O(np) time."""

    indices: list[int] = []

    for index, hyper_interval in enumerate(tree.hyper_intervals):
        if in_hyperinterval(point, hyper_interval, tree.dimension):
            indices.append(index)

    return indices

def size_segmenttree_nd(tree: SegmentTreeND):
    """Returns the total number of hyperinterval indices inside the range tree."""

    def size_segmenttree_nd_recurse(node: SegmentTreeNDNode | SegmentTreeNDNodeFinal | None) -> int:
        match node:
            case SegmentTreeNDInternalNode():
                return size_segmenttree_nd_recurse(node.left) + size_segmenttree_nd_recurse(node.right) + size_segmenttree_nd_recurse(node.internal)
            case SegmentTreeNDLeaf():
                return size_segmenttree_nd_recurse(node.internal)
            case SegmentTreeNDInternalNodeFinal():
                return size_segmenttree_nd_recurse(node.left) + size_segmenttree_nd_recurse(node.right) + len(node.internal)
            case SegmentTreeNDLeafFinal():
                return len(node.internal)
            case None:
                return 0
            
    return size_segmenttree_nd_recurse(tree.root)
