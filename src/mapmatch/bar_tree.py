""""BAR Tree is a spatial partitioning tree that recursively partitioned a point set and region that covers that point set. 
It guarantees logarithmic depth and bounded aspect ratio of all regions."""

from __future__ import annotations

import random
from typing import NamedTuple, Literal, cast, Callable
import math
import numpy

from .utilities import epsilon_leq, epsilon_geq, epsilon_equal, epsilon_range, range_closed, INFINITY, EPSILON, Interval, interval_intersect, Point

class RegionBounds(NamedTuple):
    """Bounds on the region."""
    xm: float
    xp: float
    ym: float
    yp: float
    zm: float
    zp: float

class RegionDiameter(NamedTuple):
    """Diameter of a region in x, y, x-y axis."""
    dx: float
    dy: float
    dz: float

class RegionMaximalCut(NamedTuple):
    """Maximal cut of a region."""
    Rxl: float
    Rxr: float
    Ryl: float
    Ryr: float
    Rzl: float
    Rzr: float

class RegionLargestMaximalCut(NamedTuple):
    """Maximal cut of a region where the associated region contains the most point."""
    Rx: float | None
    Ry: float | None
    Rz: float | None

class RegionSides(NamedTuple):
    """Side lengths of a region."""
    xl: float
    xr: float
    yl: float
    yr: float
    zl: float
    zr: float

class RegionVertices(NamedTuple):
    """Vertices of a region."""
    A: Point
    B: Point
    C: Point
    D: Point
    E: Point
    F: Point

class RegionTrapezoidType(NamedTuple):
    """Trapezoid properties of a region."""
    is_cit: bool
    is_crt: bool

CutAxis = Literal["x", "y", "z"]
"""Axis of a cut, which can be parallel to x, y, or x-y direction."""

class RegionCut(NamedTuple):
    """Describe a canonical cut on a region."""
    axis: CutAxis
    value: float

class RegionTwoCutInfo(NamedTuple):
    """Describe extra debugging information about the found two-cut."""
    code: str
    region_name: str
    trapezoid_type: str
    trapezoid_is_right: bool

class Region:
    """A convex region with sides parallel to the x-axis, y-axis and x-y=0 line."""

    bounds: RegionBounds
    """Bounds of the region."""

    def __init__(self, xm: float, xp: float, ym: float, yp: float, zm: float, zp: float) -> None:
        self.bounds = RegionBounds(xm, xp, ym, yp, zm, zp)

    @staticmethod
    def create_invalid_region() -> Region:
        """Creates an empty region which is a region that is invalid, i.e. has bounds with intervals [x, y] where x > y."""
        return Region(1, -1, 1, -1, 1, -1)

    @staticmethod
    def create_region_from_point_list(point: list[Point]) -> Region:
        """Creates a region from point list."""

        xm = INFINITY
        xp = -INFINITY
        ym = INFINITY
        yp = -INFINITY
        zm = INFINITY
        zp = -INFINITY

        for p in point:
            xm = min(xm, p.x)
            xp = max(xp, p.x)
            ym = min(ym, p.y)
            yp = max(yp, p.y)
            zm = min(zm, p.z)
            zp = max(zp, p.z)

        return Region(xm, xp, ym, yp, zm, zp)

    def get_reduced_region(self) -> Region:
        """Creates a new region that spans the same region geometrically, but has the smallest bounds."""
        xm, xp, ym, yp, zm, zp = self.bounds

        return Region(
            max(xm, ym + zm), min(xp, yp + zp), max(ym, xm - zp),
            min(yp, xp - zm), max(zm, xm - yp), min(zp, xp - ym)
        )

    def is_valid(self) -> bool:
        """Checks if the region has valid bounds, that is for bound [a, b] whether a <= b."""
        xm, xp, ym, yp, zm, zp = self.bounds
        return epsilon_geq(xp, xm) and epsilon_geq(yp, ym) and epsilon_geq(zp, zm)

    def is_not_singleton(self) -> bool:
        """Checks if the region is not too small where the bounds are epsilon close to each other. Topologically, it checks if the interior
        of the region is not empty."""
        xm, xp, ym, yp, zm, zp = self.bounds
        return not epsilon_equal(xm, xp) and not epsilon_equal(ym, yp) and not epsilon_equal(zm, zp)

    def get_diam_i(self) -> RegionDiameter:
        """Computes the diameter of the region in the x, y, and x-y axis in max norm."""
        xm, xp, ym, yp, zm, zp = self.bounds
        return RegionDiameter(xp - xm, yp - ym, 0.5 * (zp - zm))

    def get_diam(self) -> float:
        """Computes the global diameter of the region in max norm."""
        di = self.get_diam_i()
        return max(di)
    
    def get_min_diam(self) -> float:
        """Computes the minimum global diameter of the region in max norm."""
        di = self.get_diam_i()
        return min(di)

    def get_aspect(self) -> float:
        """Computes the aspect ratio of the region in max norm."""
        di = self.get_diam_i()
        return max(di) / min(di)

    def get_vertices(self) -> RegionVertices:
        """Computes the vertices of the region."""
        xm, xp, ym, yp, zm, zp = self.bounds
        A = Point(xm, ym)
        B = Point(zp + ym, ym)
        C = Point(xp, xp - zp)
        D = Point(xp, yp)
        E = Point(zm + yp, yp)
        F = Point(xm, xm - zm)
        return RegionVertices(A, B, C, D, E, F)

    def get_sides(self) -> RegionSides:
        """Computes the side lengths of the region in max norm."""
        [xm, xp, ym, yp, zm, zp] = self.bounds
        dy = yp - ym
        xl = (xm - zm) - ym
        xr = yp - (xp - zp)
        yl = (ym + zp) - xm
        yr = xp - (yp + zm)
        zl = dy - xl
        zr = dy - xr
        return RegionSides(xl, xr, yl, yr, zl, zr)

    def point_inside(self, point: Point, e: float=EPSILON):
        """Determines if the given point is inside the region."""
        xm, xp, ym, yp, zm, zp = self.bounds
        return epsilon_range(point.x, xm, xp, e=e) and epsilon_range(point.y, ym, yp, e=e) and epsilon_range(point.z, zm, zp, e=e)

    @staticmethod
    def get_distance(region1: Region, region2: Region) -> float:
        """Computes the distance of two regions in max norm."""
        [xm1, xp1, ym1, yp1, zm1, zp1] = region1.bounds
        [xm2, xp2, ym2, yp2, zm2, zp2] = region2.bounds
        return max(xm1 - xp2, xm2 - xp1, ym1 - yp2, ym2 - yp1, zm1 - zp2, zm2 - zp1)

class PointSet:
    """A structure of lists containing a subset of the input point set. These three all have the same points, but have different permutations,
    they are sorted in x, y, x-y in ascending order. This is useful in sweeping algorithm like finding a one-cut."""
    original_point_list: list[Point]
    permutation_sorted_x: list[int]
    permutation_sorted_y: list[int]
    permutation_sorted_z: list[int]

    def __init__(self, original_point_list: list[Point], permutation_sorted_x: list[int], permutation_sorted_y: list[int], permutation_sorted_z: list[int]) -> None:
        self.original_point_list = original_point_list
        self.permutation_sorted_x = permutation_sorted_x
        self.permutation_sorted_y = permutation_sorted_y
        self.permutation_sorted_z = permutation_sorted_z

    def get_singleton_point(self) -> Point:
        """Returns the singleton point of a point set containing one point."""
        return self.original_point_list[self.permutation_sorted_x[0]]

    def get_point_list(self) -> list[Point]:
        """Returns list of points"""
        return [self.original_point_list[ind] for ind in self.permutation_sorted_x]

    @property
    def n(self) -> int:
        """Returns the number of points the subset point set contains."""
        return len(self.permutation_sorted_x)

    @property
    def N(self) -> int:
        """Returns the number of points the original point set contains."""
        return len(self.original_point_list)

    @staticmethod
    def create_full_point_set(point_list: list[Point]) -> PointSet:
        """Create the sorted lists of the full point set."""
        numpy_point_list = numpy.array(point_list)
        permutation_sorted_x = cast(list[int], numpy.argsort(numpy_point_list[:, 0]).tolist())
        permutation_sorted_y = cast(list[int],numpy.argsort(numpy_point_list[:, 1]).tolist())
        permutation_sorted_z = cast(list[int],numpy.argsort(numpy_point_list[:, 0] - numpy_point_list[:, 1]).tolist())

        return PointSet(point_list, permutation_sorted_x, permutation_sorted_y, permutation_sorted_z)

    @staticmethod
    def create_empty_point_set(point_list: list[Point]) -> PointSet:
        """Create empty sorted lists of the full point set."""
        return PointSet(point_list, [], [], [])

class BarTreeOneCutNode(NamedTuple):
    """Describe a BAR tree node."""
    region: Region
    point_set: PointSet
    cut: RegionCut
    children: tuple[BarTreeNode, BarTreeNode]

class BarTreeTwoCutNode(NamedTuple):
    """Describe a BAR tree node."""
    region: Region
    point_set: PointSet
    cut: RegionCut
    children: tuple[BarTreeNode, BarTreeNode]

class BarTreeLeafNode(NamedTuple):
    """Describe a BAR tree node."""
    region: Region
    point_set: PointSet

BarTreeNode = BarTreeOneCutNode | BarTreeTwoCutNode | BarTreeLeafNode

class BarTreeInfo(NamedTuple):
    """Describe information about the bar tree node."""
    node_count: int
    one_cuts: int
    two_cuts: int
    leaf_count: int

def region_get_trapezoid_type(region_sides: RegionSides) -> RegionTrapezoidType:
    """Returns trapezoid property of the region, whether it is CIT or CRT."""
    xl, xr, yl, yr, zl, zr = region_sides

    number_of_sides = len([1 for ip in region_sides if not epsilon_equal(ip, 0)])
    number_of_parallel_sides = len([1 for ip, iq in [(xl, xr), (yl, yr), (zl, zr)] if not epsilon_equal(ip, 0) and not epsilon_equal(iq, 0)])
    z_are_parallel = not epsilon_equal(zl, 0) and not epsilon_equal(zr, 0)
    is_trapezoid = number_of_sides == 4 and number_of_parallel_sides == 1
    return RegionTrapezoidType(is_cit=is_trapezoid and z_are_parallel, is_crt=is_trapezoid and (not z_are_parallel))

def region_get_maximal_cut(region: Region, alpha: float) -> RegionMaximalCut:
    """Returns the maximal cut of the region, that is a cut that still make the left most region or right most region alpha-balanced."""

    def construct_sweep_points_from_candidate_event_points(events: list[float], sweep_range: Interval, reverse: bool=False) -> list[float]:
        """From a list of candidate events, create a sorted list of events that lies inside the sweep range."""

        events_inside_sweep_range = [c for c in events if epsilon_range(c, sweep_range.left, sweep_range.right)]
        return sorted(set(events_inside_sweep_range), reverse=reverse)

    region_bounds = region.bounds
    dx = Interval(region_bounds[0], region_bounds[1])
    dy = Interval(region_bounds[2], region_bounds[3])
    dz = Interval(region_bounds[4], region_bounds[5])

    candidate_events_Rxl = [region_bounds[0], region_bounds[1], region_bounds[0], region_bounds[1] + -region_bounds[3] +region_bounds[2],
             2 * region_bounds[1] + -region_bounds[3] + -region_bounds[5], region_bounds[1] + -region_bounds[5] + region_bounds[4],
             -region_bounds[1] + region_bounds[0] + region_bounds[3] + region_bounds[5],
             -2 * region_bounds[1] + 2 * region_bounds[0] + region_bounds[3] + region_bounds[5], region_bounds[2] + region_bounds[5],
             region_bounds[3] + region_bounds[5], region_bounds[3] + region_bounds[4], -region_bounds[3] + 2 * region_bounds[2] + region_bounds[5],
             region_bounds[3] + -region_bounds[5] + 2 * region_bounds[4]
            ]
    candidate_events_Rxr = [region_bounds[0], region_bounds[1], region_bounds[1], region_bounds[0] + region_bounds[3] + -region_bounds[2],
             2 * region_bounds[0] + -region_bounds[2] + -region_bounds[4], region_bounds[0] + region_bounds[5] + -region_bounds[4],
             region_bounds[1] + -region_bounds[0] + region_bounds[2] + region_bounds[4],
             2 * region_bounds[1] + -2 * region_bounds[0] + region_bounds[2] + region_bounds[4],region_bounds[3] + region_bounds[4],
             region_bounds[2] + region_bounds[4], region_bounds[2] + region_bounds[5], 2 * region_bounds[3] + -region_bounds[2] + region_bounds[4],
             region_bounds[2] + 2 * region_bounds[5] + -region_bounds[4]
            ]
    candidate_events_Ryl = [region_bounds[2], region_bounds[3], region_bounds[0] + -region_bounds[4],
             region_bounds[1] + -region_bounds[3] + region_bounds[2] + -region_bounds[4], region_bounds[1] + -region_bounds[4],
             region_bounds[1] + -region_bounds[5], -region_bounds[1] + region_bounds[0] + region_bounds[3],
             -region_bounds[1] + 2 * region_bounds[0] + -region_bounds[4], region_bounds[2],
             -region_bounds[1] + 2 * region_bounds[3] + region_bounds[4], region_bounds[3] + -region_bounds[5] + region_bounds[4],
             region_bounds[1] + -2 * region_bounds[3] + 2 * region_bounds[2] + -region_bounds[4],
             region_bounds[1] + -2 * region_bounds[5] + region_bounds[4]
            ]
    candidate_events_Ryr = [region_bounds[2], region_bounds[3], region_bounds[1] + -region_bounds[5],
             region_bounds[0] + region_bounds[3] + -region_bounds[2] + -region_bounds[5], region_bounds[0] + -region_bounds[5],
             region_bounds[0] + -region_bounds[4], region_bounds[1] + -region_bounds[0] + region_bounds[2],
             2 * region_bounds[1] + -region_bounds[0] + -region_bounds[5], region_bounds[3],
             -region_bounds[0] + 2 * region_bounds[2] + region_bounds[5], region_bounds[2] + region_bounds[5] + -region_bounds[4],
             region_bounds[0] + 2 * region_bounds[3] + -2 * region_bounds[2] + -region_bounds[5],
             region_bounds[0] + region_bounds[5] + -2 * region_bounds[4]
            ]
    candidate_events_Rzl = [region_bounds[4], region_bounds[5], region_bounds[0] + -region_bounds[2], region_bounds[1] + -region_bounds[3],
             2 * region_bounds[1] + -2 * region_bounds[2] + -region_bounds[5],
             region_bounds[1] + -region_bounds[2] + -region_bounds[5] + region_bounds[4], region_bounds[0] + -region_bounds[2],
             -2 * region_bounds[1] + 2 * region_bounds[0] + region_bounds[5], region_bounds[1] + -region_bounds[3],
             2 * region_bounds[1] + -2 * region_bounds[2] + -region_bounds[5],
             region_bounds[1] + -region_bounds[2] + -region_bounds[5] + region_bounds[4],
             -2 * region_bounds[3] + 2 * region_bounds[2] + region_bounds[5], -region_bounds[5] + 2 * region_bounds[4]
            ]
    candidate_events_Rzr = [region_bounds[4], region_bounds[5], region_bounds[1] + -region_bounds[3], region_bounds[0] + -region_bounds[2],
             2 * region_bounds[0] + -2 * region_bounds[3] + -region_bounds[4],
             region_bounds[0] + -region_bounds[3] + region_bounds[5] + -region_bounds[4], region_bounds[1] + -region_bounds[3],
             2 * region_bounds[1] + -2 * region_bounds[0] + region_bounds[4], region_bounds[0] + -region_bounds[2],
             2 * region_bounds[0] + -2 * region_bounds[3] + -region_bounds[4],
             region_bounds[0] + -region_bounds[3] + region_bounds[5] + -region_bounds[4],
             2 * region_bounds[3] + -2 * region_bounds[2] + region_bounds[4], 2 * region_bounds[5] + -region_bounds[4]
            ]

    sorted_events_Rxl = construct_sweep_points_from_candidate_event_points(candidate_events_Rxl, dx, reverse=True)
    sorted_events_Rxr = construct_sweep_points_from_candidate_event_points(candidate_events_Rxr, dx)
    sorted_events_Ryl = construct_sweep_points_from_candidate_event_points(candidate_events_Ryl, dy, reverse=True)
    sorted_events_Ryr = construct_sweep_points_from_candidate_event_points(candidate_events_Ryr, dy)
    sorted_events_Rzl = construct_sweep_points_from_candidate_event_points(candidate_events_Rzl, dz, reverse=True)
    sorted_events_Rzr = construct_sweep_points_from_candidate_event_points(candidate_events_Rzr, dz)

    def aspect_Rxl(c: float) -> Interval:
        return Interval(min(c - region_bounds[0], min(region_bounds[3], c - region_bounds[4]) - region_bounds[2], 1/2 * (min(region_bounds[5], c - region_bounds[2]) - region_bounds[4])), max(c - region_bounds[0], min(region_bounds[3], c - region_bounds[4]) - region_bounds[2], 1/2 * (min(region_bounds[5], c - region_bounds[2]) - region_bounds[4])))
    def aspect_Rxr(c: float) -> Interval:
        return Interval(min(region_bounds[1] - c, region_bounds[3] - max(region_bounds[2], c - region_bounds[5]), 1/2 * (region_bounds[5] - max(region_bounds[4], c - region_bounds[3]))), max(region_bounds[1] - c, region_bounds[3] - max(region_bounds[2], c - region_bounds[5]), 1/2 * (region_bounds[5] - max(region_bounds[4], c - region_bounds[3]))))
    def aspect_Ryl(c: float) -> Interval:
        return Interval(min(min(region_bounds[1], c + region_bounds[5]) - region_bounds[0], c - region_bounds[2], 1/2 * (region_bounds[5] - max(region_bounds[4], region_bounds[0] - c))), max(min(region_bounds[1], c + region_bounds[5]) - region_bounds[0], c - region_bounds[2], 1/2 * (region_bounds[5] - max(region_bounds[4], region_bounds[0] - c))))
    def aspect_Ryr(c: float) -> Interval:
        return Interval(min(region_bounds[1] - max(region_bounds[0], c + region_bounds[4]), region_bounds[3] - c, 1/2 * (min(region_bounds[5], region_bounds[1] - c) - region_bounds[4])), max(region_bounds[1] - max(region_bounds[0], c + region_bounds[4]), region_bounds[3] - c, 1/2 * (min(region_bounds[5], region_bounds[1] - c) - region_bounds[4])))
    def aspect_Rzl(c: float) -> Interval:
        return Interval(min(min(region_bounds[1], region_bounds[3] + c) - region_bounds[0], region_bounds[3] - max(region_bounds[2], region_bounds[0] - c), 1/2 * (c - region_bounds[4])), max(min(region_bounds[1], region_bounds[3] + c) - region_bounds[0], region_bounds[3] - max(region_bounds[2], region_bounds[0] - c), 1/2 * (c - region_bounds[4])))
    def aspect_Rzr(c: float) -> Interval:
        return Interval(min(region_bounds[1] - max(region_bounds[0], region_bounds[2] + c), min(region_bounds[3], region_bounds[1] - c) - region_bounds[2], 1/2 * (region_bounds[5] - c)), max(region_bounds[1] - max(region_bounds[0], region_bounds[2] + c), min(region_bounds[3], region_bounds[1] - c) - region_bounds[2], 1/2 * (region_bounds[5] - c)))

    def get_maximal_cut(sorted_events: list[float], aspect_from_event: Callable[[float], Interval], alpha: float) -> float:
        for c1, c2 in zip(sorted_events[:-1], sorted_events[1:]):
            min_c1, max_c1 = aspect_from_event(c1)
            min_c2, max_c2 = aspect_from_event(c2)

            if min_c2 != 0 and epsilon_leq(max_c2 / min_c2, alpha):
                continue

            denom = alpha * (min_c1 - min_c2) - (max_c1 - max_c2)
            if epsilon_equal(denom, 0):
                continue

            local_arg = (alpha * min_c1 - max_c1) / denom

            # assert epsilon_range(local_arg, 0, 1), f"Critical point must exist in interval [0, 1], it is {local_arg}"
            global_arg = c1 * (1 - local_arg) + c2 * local_arg
            return global_arg

        return sorted_events[-1]

    maximal_cut_Rxl = get_maximal_cut(sorted_events_Rxl, aspect_Rxl, alpha)
    maximal_cut_Rxr = get_maximal_cut(sorted_events_Rxr, aspect_Rxr, alpha)
    maximal_cut_Ryl = get_maximal_cut(sorted_events_Ryl, aspect_Ryl, alpha)
    maximal_cut_Ryr = get_maximal_cut(sorted_events_Ryr, aspect_Ryr, alpha)
    maximal_cut_Rzl = get_maximal_cut(sorted_events_Rzl, aspect_Rzl, alpha)
    maximal_cut_Rzr = get_maximal_cut(sorted_events_Rzr, aspect_Rzr, alpha)

    return RegionMaximalCut(maximal_cut_Rxl, maximal_cut_Rxr, maximal_cut_Ryl, maximal_cut_Ryr, maximal_cut_Rzl, maximal_cut_Rzr)

def region_find_one_cut(point_set: PointSet, Rik: RegionMaximalCut, beta: float) -> RegionCut | None:
    """Returns the one-cut of the region if it is one-cuttable, else it returns None."""
    [Rxl, Rxr, Ryl, Ryr, Rzl, Rzr] = Rik

    n = len(point_set.permutation_sorted_x)
    beta_balanced_range = range_closed(math.ceil((1 - beta) * n) - 1, math.floor(beta * n) - 1)
    for i in beta_balanced_range:
        n1 = i + 1
        n2 = n - (i + 1)

        cut_x = None
        cut_y = None
        cut_z = None

        px = point_set.original_point_list[point_set.permutation_sorted_x[i]]
        py = point_set.original_point_list[point_set.permutation_sorted_y[i]]
        pz = point_set.original_point_list[point_set.permutation_sorted_z[i]]

        qx = point_set.original_point_list[point_set.permutation_sorted_x[i+1]]
        qy = point_set.original_point_list[point_set.permutation_sorted_y[i+1]]
        qz = point_set.original_point_list[point_set.permutation_sorted_z[i+1]]

        alpha_balanced_interval_x: Interval | None = None
        alpha_balanced_interval_y: Interval | None = None
        alpha_balanced_interval_z: Interval | None = None

        # Degenerate cases when |P1|=1 or |P2|=1, then ignore alpha-balanced condition, since region will be
        # converted to singleton region anyways
        if n1 == 1 and n2 == 1:
            alpha_balanced_interval_x = Interval(-INFINITY, INFINITY)
            alpha_balanced_interval_y = Interval(-INFINITY, INFINITY)
            alpha_balanced_interval_z = Interval(-INFINITY, INFINITY)
        elif n1 == 1:
            alpha_balanced_interval_x = Interval(-INFINITY, Rxr)
            alpha_balanced_interval_y = Interval(-INFINITY, Ryr)
            alpha_balanced_interval_z = Interval(-INFINITY, Rzr)
        elif n2 == 1:
            alpha_balanced_interval_x = Interval(Rxl, INFINITY)
            alpha_balanced_interval_y = Interval(Ryl, INFINITY)
            alpha_balanced_interval_z = Interval(Rzl, INFINITY)
        else:
            alpha_balanced_interval_x = Interval(Rxl, Rxr)
            alpha_balanced_interval_y = Interval(Ryl, Ryr)
            alpha_balanced_interval_z = Interval(Rzl, Rzr)

        cut_x = interval_intersect(Interval(px.x, qx.x), alpha_balanced_interval_x)
        cut_y = interval_intersect(Interval(py.y, qy.y), alpha_balanced_interval_y)
        cut_z = interval_intersect(Interval(pz.z, qz.z), alpha_balanced_interval_z)

        if Rxl > Rxr:
            cut_x = None
        if Ryl > Ryr:
            cut_y = None
        if Rzl > Rzr:
            cut_z = None

        if cut_x is not None:
            return RegionCut("x", (cut_x[0] + cut_x[1]) / 2)
        if cut_y is not None:
            return RegionCut("y", (cut_y[0] + cut_y[1]) / 2)
        if cut_z is not None:
            return RegionCut("z", (cut_z[0] + cut_z[1]) / 2)

    # No cut is found, so the region is not one-cuttable.
    return None

def region_find_two_cut(region_diameters: RegionDiameter, region_sides: RegionSides, Rik: RegionMaximalCut, Ri: RegionLargestMaximalCut, alpha: float) -> tuple[RegionCut, RegionTwoCutInfo]:
    """Returns the two-cut of the region. This function should always output a cut since every region with correct alpha, beta ranges are two-cuttable."""

    Rxl, Rxr, Ryl, Ryr, Rzl, Rzr = Rik
    xl, xr, yl, yr, zl, zr = region_sides
    dx, dy, _ = region_diameters
    Rx_is_Rxl, Ry_is_Ryl, Rz_is_Rzl = Ri

    C = min(dx, dy) * (alpha - 2) / alpha
    is_long_rectangle = max(dx, dy) >= min(dx, dy) * (alpha + 1) / (alpha - 2)

    if dy > dx:
        if zl < C and zr < C: # both small
            assert Rz_is_Rzl is not None, "Error in case A (normal)."
            if Rz_is_Rzl:
                return (RegionCut("z", Rzl), RegionTwoCutInfo("A", "Rzl", "CIT", False))
            else:
                return (RegionCut("z", Rzr), RegionTwoCutInfo("A", "Rzr", "CIT", True))
        elif zl > C and zr > C: # both large
            if zl >= zr and yl > xl:
                assert Rx_is_Rxl is True or Ry_is_Ryl is False, "Error in case B1 (normal)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("B1", "Rxl", "CRT", False))
                else: # Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("B1", "Ryr", "CRT", True))
            elif zr >= zl and yr > xr:
                assert Rx_is_Rxl is False or Ry_is_Ryl is True, "Error in case B2 (normal)."
                if Rx_is_Rxl is False:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("B2", "Rxr", "CRT", True))
                else: # Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("B2", "Ryl", "CRT", False))
            else: # otherwise
                assert Ry_is_Ryl is not None, "Error in case C (normal)."
                if Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("C", "Ryl", "CRT", False))
                else: # Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("C", "Ryr", "CRT", True))
        elif is_long_rectangle:
            if zr <= zl:
                assert Ry_is_Ryl is False or Rz_is_Rzl is False, "Error in case D1 (normal)."
                if Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("D1", "Ryr", "CRT", True))
                else: # Rz_is_Rzl is False:
                    return (RegionCut("z", Rzr), RegionTwoCutInfo("D1", "Rzr", "CIT", True))
            else: # zr > zl
                assert Ry_is_Ryl is True or Rz_is_Rzl is True, "Error in case D2 (normal)."
                if Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("D2", "Ryl", "CRT", False))
                else: # Rz_is_Rzl is True:
                    return (RegionCut("z", Rzl), RegionTwoCutInfo("D2", "Rzl", "CIT", False))
        else: # squat rectangle
            if zl > zr:
                assert (Rx_is_Rxl is not None and Ry_is_Ryl is not None) or Rz_is_Rzl is not None, "Error in case E1/F1 (normal)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("E1", "Rxl", "CRT", False))
                elif Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("E1", "Ryr", "CRT", True))
                else:
                    return (RegionCut("z", Rzr), RegionTwoCutInfo("F1", "Rzr", "CIT", True))
            else: # zl < zr
                assert (Rx_is_Rxl is not None and Ry_is_Ryl is not None) or Rz_is_Rzl is not None, "Error in case E2/F2 (normal)."
                if Rx_is_Rxl is False:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("E2", "Rxr", "CRT", True))
                elif Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("E2", "Ryl", "CRT", False))
                else:
                    return (RegionCut("z", Rzl), RegionTwoCutInfo("F2", "Rzl", "CIT", False))
    else: # dx >= dy
        if zl < C and zr < C: # both small
            assert Rz_is_Rzl is not None, "Error in case A (flipped)."
            if Rz_is_Rzl:
                return (RegionCut("z", Rzl), RegionTwoCutInfo("fA", "Rzl", "CIT", False))
            else:
                return (RegionCut("z", Rzr), RegionTwoCutInfo("fA", "Rzr", "CIT", True))
        elif zl > C and zr > C: # both large
            if zl >= zr and xr > yr:
                assert Rx_is_Rxl is True or Ry_is_Ryl is False, "Error in case B1 (flipped)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("fB1", "Rxl", "CRT", False))
                else: # Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("fB1", "Ryr", "CRT", True))
            elif zr >= zl and xl > yl:
                assert Rx_is_Rxl is False or Ry_is_Ryl is True, "Error in case B2 (flipped)."
                if Rx_is_Rxl is False:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("fB2", "Rxr", "CRT", True))
                else: # Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("fB2", "Ryl", "CRT", False))
            else: # otherwise
                assert Rx_is_Rxl is not None, "Error in case C (flipped)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("fC", "Rxl", "CRT", False))
                else:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("fC", "Rxr", "CRT", True))
        elif is_long_rectangle:
            if zl <= zr:
                assert Rx_is_Rxl is False or Rz_is_Rzl is True, "Error in case D1 (flipped)."
                if Rx_is_Rxl is False:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("fD1", "Rxr", "CRT", True))
                else: # Rz_is_Rzl is True:
                    return (RegionCut("z", Rzl), RegionTwoCutInfo("fD1", "Rzl", "CIT", False))
            else: # zl > zr
                assert Rx_is_Rxl is True or Rz_is_Rzl is False, "Error in case D2 (flipped)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("fD2", "Rxl", "CRT", False))
                else: # Rz_is_Rzl is False:
                    return (RegionCut("z", Rzr), RegionTwoCutInfo("fD2", "Rzr", "CIT", True))
        else: # squat rectangle
            if zl < zr:
                assert (Rx_is_Rxl is not None and Ry_is_Ryl is not None) or Rz_is_Rzl is not None, "Error in case E1/F1 (flipped)."
                if Rx_is_Rxl is False:
                    return (RegionCut("x", Rxr), RegionTwoCutInfo("fE1", "Rxr", "CRT", True))
                elif Ry_is_Ryl is True:
                    return (RegionCut("y", Ryl), RegionTwoCutInfo("fE1", "Ryl", "CRT", False))
                else:
                    return (RegionCut("z", Rzl), RegionTwoCutInfo("fF1", "Rzl", "CIT", False))
            else: # zl > zr
                assert (Rx_is_Rxl is not None and Ry_is_Ryl is not None) or Rz_is_Rzl is not None, "Error in case E2/F2 (flipped)."
                if Rx_is_Rxl is True:
                    return (RegionCut("x", Rxl), RegionTwoCutInfo("fE2", "Rxl", "CRT", False))
                elif Ry_is_Ryl is False:
                    return (RegionCut("y", Ryr), RegionTwoCutInfo("fE2", "Ryr", "CRT", True))
                else:
                    return (RegionCut("z", Rzr), RegionTwoCutInfo("fF2", "Rzr", "CIT", True))

def region_make_cut(region: Region, point_set: PointSet, cut: RegionCut) -> tuple[Region, Region, PointSet, PointSet]:
    """Makes the canonical cut at the region, splitting the region and point set (R, P) to (R1, R2, P1, P2)."""

    c = cut.value
    xm, xp, ym, yp, zm, zp = region.bounds

    points_sorted_x_1: list[int] = []
    points_sorted_x_2: list[int] = []
    points_sorted_y_1: list[int] = []
    points_sorted_y_2: list[int] = []
    points_sorted_z_1: list[int] = []
    points_sorted_z_2: list[int] = []

    region1: Region | None = None
    region2: Region | None = None

    match cut.axis:
        case "x":
            assert epsilon_range(c, xm, xp), f"cut.value {c} must be in [{xm}, {xp}]."

            region1 = Region(xm, c, ym, min(yp, c - zm), zm, min(zp, c - ym))
            region2 = Region(c, xp, max(ym, c - zp), yp, max(zm, c - yp), zp)

            for point_index in point_set.permutation_sorted_x:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.x, xm, c):
                    points_sorted_x_1.append(point_index)
                if epsilon_range(p.x, c, xp):
                    points_sorted_x_2.append(point_index)
            for point_index in point_set.permutation_sorted_y:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.x, xm, c):
                    points_sorted_y_1.append(point_index)
                if epsilon_range(p.x, c, xp):
                    points_sorted_y_2.append(point_index)
            for point_index in point_set.permutation_sorted_z:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.x, xm, c):
                    points_sorted_z_1.append(point_index)
                if epsilon_range(p.x, c, xp):
                    points_sorted_z_2.append(point_index)
        case "y":
            assert epsilon_range(c, ym, yp), f"c {c} must be in [{ym}, {yp}]."

            region1 = Region(xm, min(xp, c + zp), ym, c, max(zm, xm - c), zp)
            region2 = Region(max(xm, c + zm), xp, c, yp, zm, min(zp, xp - c))

            for point_index in point_set.permutation_sorted_x:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.y, ym, c):
                    points_sorted_x_1.append(point_index)
                if epsilon_range(p.y, c, yp):
                    points_sorted_x_2.append(point_index)
            for point_index in point_set.permutation_sorted_y:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.y, ym, c):
                    points_sorted_y_1.append(point_index)
                if epsilon_range(p.y, c, yp):
                    points_sorted_y_2.append(point_index)
            for point_index in point_set.permutation_sorted_z:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.y, ym, c):
                    points_sorted_z_1.append(point_index)
                if epsilon_range(p.y, c, yp):
                    points_sorted_z_2.append(point_index)
        case "z":
            assert epsilon_range(c, zm, zp), f"c {c} must be in [{zm}, {zp}]."

            region1 = Region(xm, min(xp, yp + c), max(ym, xm - c), yp, zm, c)
            region2 = Region(max(xm, ym + c), xp, ym, min(yp, xp - c), c, zp)

            for point_index in point_set.permutation_sorted_x:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.z, zm, c):
                    points_sorted_x_1.append(point_index)
                if epsilon_range(p.z, c, zp):
                    points_sorted_x_2.append(point_index)
            for point_index in point_set.permutation_sorted_y:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.z, zm, c):
                    points_sorted_y_1.append(point_index)
                if epsilon_range(p.z, c, zp):
                    points_sorted_y_2.append(point_index)
            for point_index in point_set.permutation_sorted_z:
                p = point_set.original_point_list[point_index]
                if epsilon_range(p.z, zm, c):
                    points_sorted_z_1.append(point_index)
                if epsilon_range(p.z, c, zp):
                    points_sorted_z_2.append(point_index)

    point_set1 = PointSet(point_set.original_point_list, points_sorted_x_1, points_sorted_y_1, points_sorted_z_1)
    point_set2 = PointSet(point_set.original_point_list, points_sorted_x_2, points_sorted_y_2, points_sorted_z_2)

    return (region1, region2, point_set1, point_set2)

def region_get_largest_maximal_cut(region: Region, point_set: PointSet, Rik: RegionMaximalCut) -> RegionLargestMaximalCut:
    """Returns the maximal cut that contains the largest number of points for each axis x, y, and x-y."""

    xm, xp, ym, yp, zm, zp = region.bounds
    Rxl, Rxr, Ryl, Ryr, Rzl, Rzr = Rik

    Rxl_count = Rxr_count = Ryl_count = Ryr_count = Rzl_count = Rzr_count = 0
    Rx_is_Rxl = Ry_is_Ryl = Rz_is_Rzl = None

    for point_index in point_set.permutation_sorted_x:
        p = point_set.original_point_list[point_index]

        if epsilon_range(p.x, xm, Rxl):
            Rxl_count += 1
        if epsilon_range(p.x, Rxr, xp):
            Rxr_count += 1

        if epsilon_range(p.y, ym, Ryl):
            Ryl_count += 1
        if epsilon_range(p.y, Ryr, yp):
            Ryr_count += 1

        if epsilon_range(p.z, zm, Rzl):
            Rzl_count += 1
        if epsilon_range(p.z, Rzr, zp):
            Rzr_count += 1

    if (Rxl_count > 0 or Rxr_count > 0) and Rxl < Rxr:
        if Rxl_count >= Rxr_count:
            Rx_is_Rxl = True
        else:
            Rx_is_Rxl = False

    if (Ryl_count > 0 or Ryr_count > 0) and Ryl < Ryr:
        if Ryl_count >= Ryr_count:
            Ry_is_Ryl = True
        else:
            Ry_is_Ryl = False

    if (Rzl_count > 0 or Rzr_count > 0) and Rzl < Rzr:
        if Rzl_count >= Rzr_count:
            Rz_is_Rzl = True
        else:
            Rz_is_Rzl = False

    return RegionLargestMaximalCut(Rx_is_Rxl, Ry_is_Ryl, Rz_is_Rzl)

def region_get_maximal_cut_intersection(region: Region, Rik: RegionMaximalCut) -> list[Region]:
    """Calculates all the possible ways to make the region not one-cuttable. It does this by
    finding certain maximal cut combination that can yield R_x cap R_y cap R_z not being empty.
    This function returns a list of regions where if more than beta*n points are inside, then 
    the region is not one-cuttable. An empty list means that the region is always one-cuttable,
    for instance on CIT/CRT regions."""

    def add_candidate_region_if_valid(candidate_region: Region, intersection_list: list[Region]):
        candidate_region_reduced = candidate_region.get_reduced_region()
        if candidate_region_reduced.is_valid() and candidate_region_reduced.is_not_singleton():
            intersection_list.append(candidate_region_reduced)

    intersection_list: list[Region] = []

    ixl, ixr, iyl, iyr, izl, izr = region.bounds
    jxl, jxr, jyl, jyr, jzl, jzr = Rik

    add_candidate_region_if_valid(Region(ixl, jxl, iyl, jyl, izl, jzl), intersection_list)
    add_candidate_region_if_valid(Region(jxr, ixr, iyl, jyl, izl, jzl), intersection_list)
    add_candidate_region_if_valid(Region(ixl, jxl, jyr, iyr, izl, jzl), intersection_list)
    add_candidate_region_if_valid(Region(jxr, ixr, jyr, iyr, izl, jzl), intersection_list)
    add_candidate_region_if_valid(Region(ixl, jxl, iyl, jyl, jzr, izr), intersection_list)
    add_candidate_region_if_valid(Region(jxr, ixr, iyl, jyl, jzr, izr), intersection_list)
    add_candidate_region_if_valid(Region(ixl, jxl, jyr, iyr, jzr, izr), intersection_list)
    add_candidate_region_if_valid(Region(jxr, ixr, jyr, iyr, jzr, izr), intersection_list)

    return intersection_list

def compute_bar_tree_from_points(points: list[Point], alpha: float, beta: float) -> BarTreeNode:
    """Computes a BAR tree from an array of points."""
    region = Region.create_region_from_point_list(points)
    pointset = PointSet.create_full_point_set(points)
    return compute_bar_tree(region, pointset, alpha, beta)

def compute_bar_tree(region: Region, point_set: PointSet, alpha: float, beta: float) -> BarTreeNode:
    """Computes recursively a BAR tree of a region and point set."""

    assert region.is_valid(), "Region must stay valid."

    ERROR_MARGE = 1e-6

    match point_set.n:
        case 0:
            return BarTreeLeafNode(region, PointSet.create_empty_point_set(point_set.original_point_list))
        case 1:
            return BarTreeLeafNode(region, point_set)
        case _:
            Rik = region_get_maximal_cut(region, alpha)
            one_cut_result = region_find_one_cut(point_set, Rik, beta)

            # One-cuttable
            if one_cut_result is not None:
                region1, region2, point_set1, point_set2 = region_make_cut(region, point_set, one_cut_result)

                assert point_set1.n <= 1 or epsilon_leq(region1.get_aspect(), alpha, e=ERROR_MARGE), f"R1 must be alpha-balanced or singleton, it is {region1.get_aspect()}."
                assert point_set2.n <= 1 or epsilon_leq(region2.get_aspect(), alpha, e=ERROR_MARGE), f"R2 must be alpha-balanced or singleton, it is {region2.get_aspect()}."
                assert epsilon_leq(point_set1.n / point_set.n, beta, e=ERROR_MARGE), f"P1 must be beta-balanced, it is {point_set1.n / point_set.n} with {point_set1.n} and {point_set.n}."
                assert epsilon_leq(point_set2.n / point_set.n, beta, e=ERROR_MARGE), f"P2 must be beta-balanced, it is {point_set2.n / point_set.n} with {point_set2.n} and {point_set.n}."
                assert point_set1.n > 0 and point_set2.n > 0, "P1 and P2 must both be non zero for a one-cut."
                assert point_set1.n + point_set2.n >= point_set.n, f"P1 and P2 must cover the whole point set {point_set1.n} + {point_set2.n} >= {point_set.n}."

                node_child_1 = compute_bar_tree(region1, point_set1, alpha, beta)
                node_child_2 = compute_bar_tree(region2, point_set2, alpha, beta)
                return BarTreeOneCutNode(region, point_set, one_cut_result, (node_child_1, node_child_2))
            # Two-cuttable
            else:
                region_diameters = region.get_diam_i()
                region_sides = region.get_sides()
                Ri = region_get_largest_maximal_cut(region, point_set, Rik)

                two_cut_result, two_cut_info = region_find_two_cut(region_diameters, region_sides, Rik, Ri, alpha)
                region1, region2, point_set1, point_set2 = region_make_cut(region, point_set, two_cut_result)

                # Set R1 as CIT/CRT region for assert purposes
                if two_cut_info.trapezoid_is_right:
                    region1, region2 = region2, region1
                    point_set1, point_set2 = point_set2, point_set1

                R1_trapezoid_type = region_get_trapezoid_type(region1.get_sides())

                assert epsilon_leq(region1.get_aspect(), alpha, e=ERROR_MARGE), f"R1 must be alpha-balanced, it is {region1.get_aspect()}."
                assert epsilon_leq(region2.get_aspect(), alpha, e=ERROR_MARGE), f"R2 must be alpha-balanced, it is {region2.get_aspect()}."
                assert epsilon_geq(point_set1.n / point_set.n, beta, e=ERROR_MARGE), f"R1 must not be beta-balanced, it is {point_set1.n / point_set.n}."
                assert epsilon_leq(point_set2.n / point_set.n, beta, e=ERROR_MARGE), f"P2 must be beta-balanced, it is {point_set2.n / point_set.n}."
                assert point_set1.n + point_set2.n >= point_set.n, f"P1 and P2 must cover the whole point set {point_set1.n} + {point_set2.n} >= {point_set.n}."

                assert R1_trapezoid_type.is_cit or R1_trapezoid_type.is_crt and not (R1_trapezoid_type.is_cit and R1_trapezoid_type.is_crt), "R1 must be either CIT or CRT."

                if two_cut_info.trapezoid_type == "CIT":
                    assert R1_trapezoid_type.is_cit, "R1 must be CIT."
                elif two_cut_info.trapezoid_type == "CRT":
                    assert R1_trapezoid_type.is_crt, "R1 must be CRT."

                # Reswap R1 as as original position, to maintain correct position for query purposes
                if two_cut_info.trapezoid_is_right:
                    region1, region2 = region2, region1
                    point_set1, point_set2 = point_set2, point_set1

                node_child_1 = compute_bar_tree(region1, point_set1, alpha, beta)
                node_child_2 = compute_bar_tree(region2, point_set2, alpha, beta)
                return BarTreeTwoCutNode(region, point_set, two_cut_result, (node_child_1, node_child_2))

def bar_tree_info(node: BarTreeNode) -> BarTreeInfo:
    """Get information about the BAR tree."""
    match node:
        case BarTreeOneCutNode():
            left: BarTreeInfo = bar_tree_info(node.children[0])
            right: BarTreeInfo = bar_tree_info(node.children[1])

            return BarTreeInfo(
                node_count= 1 + left.node_count + right.node_count,
                one_cuts=   1 + left.one_cuts + right.one_cuts,
                two_cuts=   left.two_cuts + right.two_cuts,
                leaf_count= left.leaf_count + right.leaf_count
            )
        case BarTreeTwoCutNode():
            left: BarTreeInfo = bar_tree_info(node.children[0])
            right: BarTreeInfo = bar_tree_info(node.children[1])

            return BarTreeInfo(
                node_count= 1 + left.node_count + right.node_count,
                one_cuts=   left.one_cuts + right.one_cuts,
                two_cuts=   1 + left.two_cuts + right.two_cuts,
                leaf_count= left.leaf_count + right.leaf_count
            )
        case BarTreeLeafNode():
            return BarTreeInfo(
                node_count= 1, 
                one_cuts=   0,
                two_cuts=   0,
                leaf_count= 1
            )

def generate_random_region_point(points_count: int, point_sample_radius: float, alpha: float) -> list[Point]:
    """Generate general position satisfying point set."""
    region: Region | None = None
    Rik_intersection: Region | None = None

    while region is None or Rik_intersection is None:
        xm = random.uniform(-point_sample_radius, point_sample_radius)
        xp = random.uniform(xm, point_sample_radius)
        ym = random.uniform(-point_sample_radius, point_sample_radius)
        yp = random.uniform(ym, point_sample_radius)
        zm = random.uniform(-point_sample_radius, point_sample_radius)
        zp = random.uniform(zm, point_sample_radius)

        region = Region(xm, xp, ym, yp, zm, zp)
        region = region.get_reduced_region()

        if not region.is_valid() or not region.is_not_singleton() or region.get_aspect() > alpha or region.get_min_diam() < 0.1:
            continue

        Rik_intersections = region_get_maximal_cut_intersection(region, region_get_maximal_cut(region, alpha))
        if len(Rik_intersections) == 0:
            continue
        Rik_intersection = Rik_intersections[random.randint(0, len(Rik_intersections) - 1)]

    points: list[Point] = []
    for p in region.get_vertices():
        close_global = False
        for q in points:
            if max(abs(p.x - q.x), abs(p.y - q.y), abs(p.z - q.z)) < EPSILON * 100:
                close_global = True
                break
        if not close_global:
            pp = p

            close = True

            while close:
                pp = Point(pp.x + random.uniform(-10 * EPSILON, 10 * EPSILON), pp.y + random.uniform(-10 * EPSILON, 10 * EPSILON))

                close = False

                for q in points:
                    if min(abs(pp.x - q.x), abs(pp.y - q.y), abs(pp.z - q.z)) < EPSILON * 100:
                        close = True
                        break

                if not close:
                    points.append(pp)

    fill_bounds = Rik_intersection.bounds

    for _ in range(points_count - len(points)):
        close = True
        p: Point | None = None

        while close:
            close = False

            py = random.uniform(fill_bounds.ym, fill_bounds.yp)
            px = random.uniform(max(fill_bounds.xm, py + fill_bounds.zm), min(fill_bounds.xp, py + fill_bounds.zp))
            p = Point(px, py)

            for q in points:
                if min(abs(p.x - q.x), abs(p.y - q.y), abs(p.z - q.z)) < EPSILON * 100:
                    close = True
                    break

        assert p is not None, "Random point should be generated at least once."
        points.append(p)

    return points
