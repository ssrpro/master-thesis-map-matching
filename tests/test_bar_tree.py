import math

from mapmatch.utilities import Point, epsilon_equal
from mapmatch.bar_tree import Region, RegionMaximalCut, RegionVertices, RegionDiameter, RegionSides, RegionTrapezoidType, region_get_trapezoid_type, \
    region_get_maximal_cut, region_get_maximal_cut_intersection, generate_random_region_point, compute_bar_tree, PointSet, BarTreeNode, BarTreeLeafNode, \
    BarTreeOneCutNode, BarTreeTwoCutNode

def epsilon_equal_region_max_cut(max_cut1: RegionMaximalCut, max_cut2: RegionMaximalCut) -> bool:
    return epsilon_equal(0, max(
        abs(max_cut1.Rxl - max_cut2.Rxl),
        abs(max_cut1.Rxr - max_cut2.Rxr),
        abs(max_cut1.Ryl - max_cut2.Ryl),
        abs(max_cut1.Ryr - max_cut2.Ryr),
        abs(max_cut1.Rzl - max_cut2.Rzl),
        abs(max_cut1.Rzr - max_cut2.Rzr),
    ))

def test_standard_region():
    """Standard region"""
    region = Region(0, 3, 0, 3, -2, 2)

    assert epsilon_equal(region.get_aspect(), 3/2)
    assert region.get_sides() == RegionSides(2, 2, 2, 2, 1, 1)
    assert region.get_diam_i() == RegionDiameter(3, 3, 2)
    assert region.get_reduced_region().bounds == region.bounds
    assert region.get_vertices() == RegionVertices(A=Point(0, 0), B=Point(2, 0), C=Point(3, 1), D=Point(3, 3), E=Point(1, 3), F=Point(0, 2))
    assert region.get_diam() == 3
    assert region.is_valid()
    assert region.is_not_singleton()
    assert region_get_trapezoid_type(region.get_sides()) == RegionTrapezoidType(is_cit=False, is_crt=False)

    assert epsilon_equal_region_max_cut(region_get_maximal_cut(region, 5), RegionMaximalCut(Rxl=0.5, Rxr=2.5, Ryl=0.5, Ryr=2.5, Rzl=-4/3, Rzr=4/3))
    assert len(region_get_maximal_cut_intersection(region, region_get_maximal_cut(region, 5))) == 0

    assert epsilon_equal_region_max_cut(region_get_maximal_cut(region, 4), RegionMaximalCut(Rxl=2/3, Rxr=3-2/3, Ryl=2/3, Ryr=3-2/3, Rzl=-1, Rzr=1))
    assert len(region_get_maximal_cut_intersection(region, region_get_maximal_cut(region, 4))) > 0

def test_trapezium():
    crt_region = Region(0, 1, 0, 3, -2, 1)
    parallellogram_region = Region(0, 1, 0, 3, -2, 0)
    cit_region = Region(0, 2, 1, 3, -2, -1)

    assert region_get_trapezoid_type(cit_region.get_sides()) == RegionTrapezoidType(is_cit=True, is_crt=False)
    assert region_get_trapezoid_type(parallellogram_region.get_sides()) == RegionTrapezoidType(is_cit=False, is_crt=False)
    assert region_get_trapezoid_type(crt_region.get_sides()) == RegionTrapezoidType(is_cit=False, is_crt=True)

def test_region_distance():
    region1 = Region(0, 1, -1, 0, 0, 2)
    region2 = Region(3, 4, -1, 0, 3, 4)

    assert region1.get_diam() == 1
    assert region2.get_diam() == 1
    assert Region.get_distance(region1, region2) == 2

def test_two_cut():
    K = 16
    N = 32
    D = 4
    alpha = 6
    beta = 2/3

    for _ in range(K):
        points = generate_random_region_point(N, D, alpha)
        region = Region.create_region_from_point_list(points)
        compute_bar_tree(region, PointSet.create_full_point_set(points), alpha, beta)

def test_bar_tree_query():
    N = 128
    D = 4
    alpha = 6
    beta = 2/3
    points = generate_random_region_point(N, D, alpha)
    region = Region.create_region_from_point_list(points)
    tree = compute_bar_tree(region, PointSet.create_full_point_set(points), alpha, beta)

    for p in points:
        step_count = 0
        query_node: BarTreeNode = tree

        while True:
            match query_node:
                case BarTreeOneCutNode() | BarTreeTwoCutNode():
                    assert query_node.region.point_inside(p)
                    dist_R12 = Region.get_distance(query_node.children[0].region, query_node.children[1].region)
                    assert epsilon_equal(dist_R12, 0)

                    choose_left = None
                    match query_node.cut.axis:
                        case "x":
                            choose_left = p.x < query_node.cut.value
                        case "y":
                            choose_left = p.y < query_node.cut.value
                        case "z":
                            choose_left = p.z < query_node.cut.value

                    assert choose_left is not None

                    if choose_left:
                        query_node = query_node.children[0]
                    else:
                        query_node = query_node.children[1]

                    step_count += 1
                case BarTreeLeafNode():
                    assert query_node.point_set.n == 1
                    assert query_node.point_set.get_point_list()[0] == p
                    assert step_count <= 2 * math.log(1/query_node.point_set.N, beta)
                    break
