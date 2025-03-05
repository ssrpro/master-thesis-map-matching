import random

from mapmatch.box_tree import create_rangetree_nd, query_rangetree_nd, query_rangetree_nd_naive, \
    create_segmenttree_nd, query_segmenttree_nd, query_segmenttree_nd_naive, create_segmenttree_1d, query_segmenttree_1d, query_segmenttree_1d_naive
from mapmatch.utilities import PointND, HyperInterval, Interval

def test_rangetree():
    point_count = 256
    point_sample_radius = 4
    dimension = 5

    points: list[PointND] = []

    for _ in range(point_count):
        point_list: list[float] = []
        for _ in range(dimension):
            point_list.append(random.uniform(-point_sample_radius, point_sample_radius))

        point = tuple(point_list)
        points.append(point)

    tree = create_rangetree_nd(points, dimension)

    boxes: list[HyperInterval] = []
    for _ in range(point_count):
        interval_list: list[Interval] = []
        for _ in range(dimension):
            a = random.uniform(-point_sample_radius, point_sample_radius)
            b = random.uniform(-point_sample_radius, point_sample_radius)
            interval_list.append(Interval(min(a, b), max(a, b)))

        box = tuple(interval_list)
        boxes.append(box)

    for box in boxes:
        result_range_tree = query_rangetree_nd(tree, box)
        result_naive = query_rangetree_nd_naive(tree, box)
        assert set(result_range_tree) == set(result_naive)

def test_segmenttree_1d():
    point_count = 512
    point_sample_radius = 4

    boxes: list[Interval] = []
    for _ in range(point_count):
        a = random.uniform(-point_sample_radius, point_sample_radius)
        b = random.uniform(-point_sample_radius, point_sample_radius)
        boxes.append(Interval(min(a, b), max(a, b)))

    tree = create_segmenttree_1d(boxes)

    points: list[float] = []

    for _ in range(point_count):
        points.append(random.uniform(-point_sample_radius, point_sample_radius))

    for point in points:
        result_range_tree = query_segmenttree_1d(tree, point)
        result_naive = query_segmenttree_1d_naive(tree, point)
        assert set(result_range_tree) == set(result_naive)

def test_segmenttree():
    point_count = 256
    point_sample_radius = 4
    dimension = 5

    boxes: list[HyperInterval] = []
    for _ in range(point_count):
        interval_list: list[Interval] = []
        for _ in range(dimension):
            a = random.uniform(-point_sample_radius, point_sample_radius)
            b = random.uniform(-point_sample_radius, point_sample_radius)
            interval_list.append(Interval(min(a, b), max(a, b)))

        box = tuple(interval_list)
        boxes.append(box)

    tree = create_segmenttree_nd(boxes, dimension)

    points: list[PointND] = []

    for _ in range(point_count):
        point_list: list[float] = []
        for _ in range(dimension):
            point_list.append(random.uniform(-point_sample_radius, point_sample_radius))

        point = tuple(point_list)
        points.append(point)

    for point in points:
        result_range_tree = query_segmenttree_nd(tree, point)
        result_naive = query_segmenttree_nd_naive(tree, point)
        assert set(result_range_tree) == set(result_naive)
