"""This module contains general purpose utility functions, including floating point equality and inequality and datastructres like Point
used all over the program."""

import math
import os
import pickle
from typing import Any, NamedTuple

import numpy as np

EPSILON = 1e-10
"""Small constant used for equality comparison on floating point number on accumulated rounding errors."""

INFINITY = 1e10
"""A big constant that is practically larger than any number."""

class Point(NamedTuple):
    """A point in 2d."""
    x: float
    y: float

    @property
    def z(self):
        """The BAR tree defined z component, which is equal to x-y"""
        return self.x - self.y

class Point3D(NamedTuple):
    """A point in 3d."""
    x: float
    y: float
    z: float

PointND = tuple[float, ...]
"""A point in nd."""

class Interval(NamedTuple):
    """Store interval endpoints."""
    left: float
    right: float

class Rectangle(NamedTuple):
    """Store 2 dimensional hyperrectangle."""
    x: Interval
    y: Interval

class Cuboid(NamedTuple):
    """Store 3 dimensional hyperrectangle."""
    x: Interval
    y: Interval
    z: Interval

HyperInterval = tuple[Interval, ...]
"""Store n dimensional hyperrectangle."""

class IntRange(NamedTuple):
    """Represents a range including begin, and excluding end."""
    begin: int
    end: int

class FloatListDistribution(NamedTuple):
    """Contains statistics aboout float list."""
    mean: float
    std_dev: float
    min_value: float
    q1: float
    q2: float
    q3: float
    max_value: float

    def __str__(self) -> str:
        return f"avg {self.mean} ~{self.std_dev} :: ({self.min_value, self.q1, self.q2, self.q3, self.max_value})"

def read_pickle(name: str) -> Any:
    """"Read pickle resource."""
    try:
        with open(os.path.join("output", f"{name}.bin"), "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Can not read pickle resource {name}. Please run previous steps before executing this step.")

def get_distribution(data: list[float]) -> FloatListDistribution:
    """Calculate statistics on a float list."""
    data_sorted = sorted(data)
    q1 = np.percentile(data_sorted, 25)
    q2 = np.median(data_sorted)
    q3 = np.percentile(data_sorted, 75)
    mean = np.mean(data_sorted)
    std_dev = np.std(data_sorted)
    min_value = np.min(data_sorted)
    max_value = np.max(data_sorted)
    return FloatListDistribution(
        mean=mean.item(),
        std_dev=std_dev.item(),
        min_value=min_value.item(),
        q1=q1.item(),
        q2=q2.item(),
        q3=q3.item(),
        max_value=max_value.item()
    )

def epsilon_equal(x: float, y: float, e: float=EPSILON):
    """Check whether x and y are equal within epsilon range."""
    return abs(x - y) < e

def epsilon_range(x: float, a: float, b: float, e: float=EPSILON):
    """Check whether x is in closed interval [a, b], where equality is within epsilon range."""
    return (x > a or epsilon_equal(x, a, e=e)) and (x < b or epsilon_equal(x, b, e=e))

def integer_range(x: int, a: int, b: int):
    """Check whether x is in [a, b] for integer."""
    return (x >= a) and (x <= b)

def epsilon_leq(x: float, y: float, e: float=EPSILON):
    """Check whether x <= y, where equality is within epsilon range."""
    return x < y or epsilon_equal(x, y, e)

def epsilon_geq(x: float, y: float, e: float =EPSILON):
    """Check whether x >= y, where equality is within epsilon range."""
    return x > y or epsilon_equal(x, y, e)

def epsilon_open_range(x: float, a: float, b: float):
    """Check whether x is in open interval (a, b), where not-equality is within epsilon range."""
    return (x > a and not epsilon_equal(x, a)) and (x < b and not epsilon_equal(x, b))

def range_closed(a: int, b: int) -> range:
    """Gives an integer iterator from [a, b] where a and b are included."""
    return range(a, b+1)

def lerp(x: float, a: float, b: float) -> float:
    """Linear interpolate between two numbers."""
    return (1 - x) * a + x * b

def lerp_point(a: Point, b: Point, t: float) -> Point:
    """Linear interpolate between two points."""
    return Point((1 - t) * a.x + t * b.x, (1 - t) * a.y + t * b.y)

def clamp(x: float, interval: Interval) -> float:
    """Clamps value x within the interval range"""
    return max(interval.left, min(interval.right, x))

def solve_quadratic_equation(A: float, B: float, C: float) -> tuple[float, float] | None:
    """Solve quadratic equation Axx + Bx + C = 0 as a pair. If there are no roots, then None will be returned."""
    D = B ** 2 - 4 * A * C

    if D >= 0 and not epsilon_equal(A, 0):
        DD = math.sqrt(D)
        r1, r2 = ((-B - DD) / (2 * A), (-B + DD) / (2 * A))
        return (min(r1, r2), max(r1, r2))
    else:
        return None

def line_point_distance(l1: Point, l2: Point, p: Point) -> float:
    """Calculates the length between a line l1l2 segment and point p."""
    a = Point(p.x - l1.x, p.y - l1.y)
    b = Point(l2.x - l1.x, l2.y - l1.y)

    assert not epsilon_equal(b.x ** 2 + b.y ** 2, 0), "Given segment must have non zero length." 

    projection_scalar = (a.x * b.x + a.y * b.y) / (b.x ** 2 + b.y ** 2)
    if projection_scalar < 0:
        return point_point_distance(l1, p)
    elif projection_scalar > 1:
        return point_point_distance(l2, p)
    else: # 0 <= projection_scalar <= 1
        lp = Point(l1.x + b.x * projection_scalar, l1.y + b.y * projection_scalar)
        return point_point_distance(lp, p)

def point_point_distance(p1: Point, p2: Point) -> float:
    """Calculate the distance between two points."""
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

def interval_intersect(interval1: Interval | None, interval2: Interval | None, e: float = EPSILON) -> Interval | None:
    """Returns the intersection between two intervals where either of them may be None."""

    if interval1 is None or interval2 is None:
        return None

    result = Interval(max(interval1.left, interval2.left), min(interval1.right, interval2.right))
    if interval_valid(result, e):
        return result
    else:
        return None

def interval_mid(interval: Interval | None) -> float | None:
    """Return the mid value of an interval."""
    if interval is not None:
        return 0.5 * (interval.left + interval.right)
    else:
        return None

def interval_valid(interval: Interval, e: float = EPSILON) -> bool:
    """Checks for interval [a, b] whether a <= b with EPSILON relaxation."""
    return interval.left - e < interval.right + e

def in_interval(value: float, bound: Interval, e: float = EPSILON) -> bool:
    """Determines whether a value lies inside a interval."""
    return (bound.left < value and value < bound.right) or epsilon_equal(bound.left, value, e) or epsilon_equal(bound.right, value, e)

def in_rectangle(value: Point, bound: Rectangle, e: float = EPSILON) -> bool:
    """Determines whether a 2d point lies inside a rectangle."""
    return in_interval(value.x, bound.x, e) and in_interval(value.y, bound.y, e)

def in_cuboid(value: Point3D, bound: Cuboid, e: float = EPSILON) -> bool:
    """Determines whether a 3d point lies inside a rectangular cuboid."""
    return in_interval(value.x, bound.x, e) and in_interval(value.y, bound.y, e) and in_interval(value.z, bound.z, e)

def in_hyperinterval(point: PointND, hyper_interval: HyperInterval, dimension: int) -> bool:
    """Determines whether a nd point lies inside a hyperinterval."""
    for i in range(dimension):
        if not in_interval(point[i], hyper_interval[i]):
            return False
    return True

def circle_ab_segment_intersection(circle_origin: Point, circle_radius: float, a: Point, b: Point) -> tuple[float, float] | None:
    """Calculate the intersection between circle and segment ab as a pair. If not intersection is found, then None is returned."""
    A = (b.x - a.x) ** 2 + (b.y - a.y) ** 2
    B = 2 * ((b.x - a.x) * (a.x - circle_origin.x) + (b.y - a.y) * (a.y - circle_origin.y))
    C = (a.x - circle_origin.x) ** 2 + (a.y - circle_origin.y) ** 2 - circle_radius ** 2
    intersection_ts = solve_quadratic_equation(A, B, C)

    if intersection_ts is None:
        return None
    else:
        t1 = intersection_ts[0]
        t2 = intersection_ts[1]
        return (t1, t2)

def line_line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> tuple[float, float] | None:
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    (x4, y4) = p4
    
    num1 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    num2 = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if epsilon_equal(denom, 0):
        return None

    return num1 / denom, num2 / denom

def line_mid_point(p1: Point, p2: Point) -> Point:
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
