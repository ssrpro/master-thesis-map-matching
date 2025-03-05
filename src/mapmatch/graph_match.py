"""Describes datastructures for matching close enough points at graph vertices or points on graph edges."""

import math
from typing import NamedTuple, cast

from .input_graph import Graph, GraphNeighbour
from .utilities import Interval, Rectangle, Point, Point3D, INFINITY, in_rectangle, Cuboid, point_point_distance, \
    line_point_distance, interval_intersect, epsilon_geq, epsilon_leq, lerp_point, epsilon_equal, PointND, HyperInterval, \
    in_interval
from .box_tree import RangeTreeND, create_rangetree_nd, query_rangetree_nd, SegmentTreeND, create_segmenttree_nd, query_segmenttree_nd

class KCenter(NamedTuple):
    """Datastructure for K-center problem. It has a vertex index of a certain graph, with the radius, which
    is the largest distance between any point with this and their previous K-centers."""
    vertex: int
    radius: float
    position: int

class GraphMatch(NamedTuple):
    """
    Main datastructure for matching any p point to another point q that is either a vertex or inside an edge in the graph.
    Since we assume the graph is connected, there always exists two points that are the endpoints of the edge, or matched vertex
    index with another vertex along an edge.
    """
    point: Point
    edge_1: int
    edge_2: int

def gonzales_clustering(graph: Graph, distance_matrix: list[float]) -> list[KCenter]:
    """Gonzales clustering algorithm is a 2-approxiomation algorithm for K-center problem. In this setting
    it is used in Graph on the shortest vertex distance instead of point set and their straight line distance."""
    clustering: list[KCenter] = []
    V = len(graph.vertices)
    d: list[float] = []

    for _ in range(V):
        d.append(INFINITY)

    next_u = 0

    for _ in range(V):
        u = next_u

        for v in range(V):
            d[v] = min(d[v], distance_matrix[v + V * u])

        max_di = -1
        max_di_arg = None

        for i, di in enumerate(d):
            if di > max_di:
                max_di = di
                max_di_arg = i

        assert max_di_arg is not None

        next_u = max_di_arg
        u_radius = max(d)

        index = len(clustering)
        clustering.append(KCenter(u, u_radius, index))

    return clustering

class VertexMatchStructure(NamedTuple):
    """Datastructure to find close enough point to a vertex of the graph."""
    tree: RangeTreeND
    kcenter_list: list[KCenter]
    kcenter_points: list[Point3D]
    epsilon: float

def construct_vertex_match(graph: Graph, distance_matrix: list[float], epsilon: float) -> VertexMatchStructure:
    """Compute vertex match structure with epsilon accuracy."""
    kcenter_list = gonzales_clustering(graph, distance_matrix)
    kcenter_points: list[Point3D] = []

    for vertex_index, radius, _ in kcenter_list:
        x, y = graph.vertices[vertex_index]
        kcenter_points.append(Point3D(x, y, radius))

    tree = create_rangetree_nd(cast(list[PointND], kcenter_points), 3)
    return VertexMatchStructure(tree, kcenter_list, kcenter_points, epsilon)

def query_vertex_match(graph: Graph, graph_neighbour: GraphNeighbour, vertex_match: VertexMatchStructure, square_origin: Point, square_radius: float) -> list[GraphMatch]:
    """Query vertex match structure with epsilon accuracy. This is similar to T2 in the paper."""
    Sp = Rectangle(
        Interval(square_origin.x - 2 * square_radius, square_origin.x + 2 * square_radius),
        Interval(square_origin.y - 2 * square_radius, square_origin.y + 2 * square_radius)
    )

    bound = Cuboid(Sp.x, Sp.y, Interval(vertex_match.epsilon * square_radius, INFINITY))

    indices = query_rangetree_nd(vertex_match.tree, bound)
    T1_kcenter = [vertex_match.kcenter_list[index] for index in set(indices)]

    T1: list[GraphMatch] = []

    for kcenter in T1_kcenter:
        edge_1 = kcenter.vertex
        edge_2 = graph_neighbour.get_vertex_neighbours(kcenter.vertex)[0]
        point = graph.vertices[kcenter.vertex]

        assert epsilon_geq(kcenter.radius, vertex_match.epsilon * square_radius), "Invariant for points in T1; see Lemma 18 in cpacked"
        T1.append(GraphMatch(point, edge_1, edge_2))

    ip1 = 0
    if len(T1_kcenter) > 0:
        ip1 = max([p.position for p in T1_kcenter]) + 1
    last_vertex_kcenter = vertex_match.kcenter_list[ip1]
    last_vertex_index = vertex_match.kcenter_list[ip1].vertex

    T2 = list(T1)
    if in_rectangle(graph.vertices[last_vertex_index], Sp):
        edge_1 = last_vertex_index
        edge_2 = graph_neighbour.get_vertex_neighbours(last_vertex_index)[0]

        assert epsilon_leq(last_vertex_kcenter.radius, vertex_match.epsilon * square_radius), "Invariant extra point in T2; see Lemma 17 in cpacked"
        T2.append(GraphMatch(graph.vertices[last_vertex_index], edge_1, edge_2))

    return T2

class Cube(NamedTuple):
    """Special case of cuboid where side length of every component are equal, defined as side_length."""
    cuboid: Cuboid
    side_length: float

class Through(NamedTuple):
    """
    3D convex set made from union of triangular prism and two half-cones. It is called through in the paper.
    The set is defined as { sqrt[(x - e.x)^2 + (y - e.y)^2] <= 4z <= 8|e|eps^-1 }.
    """
    a: Point
    b: Point
    a_index: int
    b_index: int
    epsilon: float

class ThroughStructure(NamedTuple):
    """A datastructure to efficiently query a through object from a point query in O(log^3 n + kn) time."""
    through: Through
    bounding_cube: Cube
    neighbours: list[Through]

class Circle(NamedTuple):
    """Represents a 2D circle using origin and radius."""
    origin: Point
    radius: float

class RotatedRectangle(NamedTuple):
    """
    Represents a 2D rotated rectangle using 4 points a, b, c, d. The shape is made from segments ab, bc, cd, da.
    """
    a: Point
    b: Point
    c: Point
    d: Point

class EdgeMatchStructure(NamedTuple):
    """Datastructure to find close enough point p to a point q in an edge of the graph."""
    vertex_match: VertexMatchStructure
    structure: list[ThroughStructure]
    tree: SegmentTreeND
    cuboids: list[Cuboid]
    epsilon: float

class EdgeMatchQueryInfo(NamedTuple):
    """Represents information about the size of the edge match query."""
    vertex_match_count: int
    edge_match_count: int
    total_match_count: int
    trough_count: int
    avg_trough_subdivision: float

def cuboid_to_cube(cuboid: Cuboid) -> Cube:
    """Construct the smallest cube containing the input cuboid."""
    xm = 0.5 * (cuboid.x.left + cuboid.x.right)
    ym = 0.5 * (cuboid.y.left + cuboid.y.right)
    zm = 0.5 * (cuboid.z.left + cuboid.z.right)
    side_radius = 0.5 * max(cuboid.x.right - cuboid.x.left, cuboid.y.right - cuboid.y.left, cuboid.z.right - cuboid.z.left)
    cube_cuboid = Cuboid(Interval(xm - side_radius, xm + side_radius), Interval(ym - side_radius, ym + side_radius), Interval(zm - side_radius, zm + side_radius))
    return Cube(cube_cuboid, 2 * side_radius)

def circle_rectangle_intersect(circle: Circle, rectangle: Rectangle) -> bool:
    """Decide whether a given circle and rectangle intersect."""
    dx = circle.origin.x - max(rectangle.x.left, min(circle.origin.x, rectangle.x.right))
    dy = circle.origin.y - max(rectangle.y.left, min(circle.origin.y, rectangle.y.right))
    return dx ** 2 + dy ** 2 < circle.radius ** 2

def project_point_p_to_segment_ab(p: Point, a: Point, b: Point) -> float:
    """Project a point p to a t-parametrized line segment ab as p', where t in [0, 1] if and only if p' in ab."""
    return ((p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y)) / ((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

def rot_rectangle_rectangle_intersect(rotated_rectangle: RotatedRectangle, rectangle: Rectangle) -> bool:
    """Decide whether a given rotated rectangle and axis aligned rectangle intersect."""
    a, b, c, d = rotated_rectangle

    e = Point(rectangle.x.left, rectangle.y.right)
    f = Point(rectangle.x.right, rectangle.y.right)
    g = Point(rectangle.x.right, rectangle.y.left)
    h = Point(rectangle.x.left, rectangle.y.left)

    proj_ab = [
        project_point_p_to_segment_ab(e, a, b),
        project_point_p_to_segment_ab(f, a, b),
        project_point_p_to_segment_ab(g, a, b),
        project_point_p_to_segment_ab(h, a, b)
    ]

    proj_ad = [
        project_point_p_to_segment_ab(e, a, d),
        project_point_p_to_segment_ab(f, a, d),
        project_point_p_to_segment_ab(g, a, d),
        project_point_p_to_segment_ab(h, a, d)
    ]

    proj_ef = [
        project_point_p_to_segment_ab(a, e, f),
        project_point_p_to_segment_ab(b, e, f),
        project_point_p_to_segment_ab(c, e, f),
        project_point_p_to_segment_ab(d, e, f)
    ]

    proj_eh = [
        project_point_p_to_segment_ab(a, e, h),
        project_point_p_to_segment_ab(b, e, h),
        project_point_p_to_segment_ab(c, e, h),
        project_point_p_to_segment_ab(d, e, h)
    ]

    return min(max(proj_ab), max(proj_ad), max(proj_ef), max(proj_eh)) <= 0 or max(min(proj_ab), min(proj_ad), min(proj_ef), min(proj_eh)) >= 1

def through_cuboid_intersect(through: Through, cuboid: Cuboid) -> bool:
    """Decide whether a given through object and cuboid intersect."""
    through_ab_length = point_point_distance(through.a, through.b)
    through_interval = Interval(0, 2 / (through.epsilon ** 2) * (through_ab_length ** 2))
    z_interval = interval_intersect(through_interval, cuboid.z)
    if z_interval is None:
        return False

    z = z_interval.right
    circle_a = Circle(through.a, 4 * z)
    circle_b = Circle(through.b, 4 * z)

    perp_ab_x = through.b.y - through.a.y / through_ab_length
    perp_ab_y = -(through.b.x - through.a.x) / through_ab_length
    rot_rectangle_ab = RotatedRectangle(
        Point(through.a.x + perp_ab_x * 4 * z, through.a.y + perp_ab_y * 4 * z),
        Point(through.b.x + perp_ab_x * 4 * z, through.b.y + perp_ab_y * 4 * z),
        Point(through.b.x - perp_ab_x * 4 * z, through.b.y - perp_ab_y * 4 * z),
        Point(through.a.x - perp_ab_x * 4 * z, through.a.y - perp_ab_y * 4 * z)
    )

    rectangle_cuboid = Rectangle(cuboid.x, cuboid.y)
    return circle_rectangle_intersect(circle_a, rectangle_cuboid) or circle_rectangle_intersect(circle_b, rectangle_cuboid) or rot_rectangle_rectangle_intersect(rot_rectangle_ab, rectangle_cuboid)

def through_point_intersect(through: Through, point: Point3D) -> bool:
    """Decide whether a given through object and point intersect."""
    point_xy = Point(point.x, point.y)
    return 4 * point.z <= 8 / through.epsilon * point_point_distance(through.a, through.b) and line_point_distance(through.a, through.b, point_xy) <= 4 * point.z

def through_bounds(through: Through) -> Cuboid:
    """Computes the smallest cuboid containing the given through object."""
    a, b, _, _, epsilon = through
    through_ab_length = point_point_distance(a, b)
    fac = 8 / epsilon * through_ab_length
    return Cuboid(Interval(min(a.x, b.x) - fac, max(a.x, b.x) + fac), Interval(min(a.y, b.y) - fac, max(a.y, b.y) + fac), Interval(0, fac / 4))

def calculate_through_structure(throughs: list[Through]) -> list[ThroughStructure]:
    """Calculate helper structure to efficiently point query through objects."""
    cubes: list[Cube] = []

    for through in throughs:
        bounds = through_bounds(through)
        cubes.append(cuboid_to_cube(bounds))

    result: list[ThroughStructure] = []

    for through1, cube1 in zip(throughs, cubes):
        neighbours: list[Through] = []

        for through2, cube2 in zip(throughs, cubes):
            if epsilon_geq(cube2.side_length, cube1.side_length) and through_cuboid_intersect(through2, cube1.cuboid):
                neighbours.append(through2)

        result.append(ThroughStructure(through1, cube1, neighbours))

    return result

def construct_edge_match(graph: Graph, distance_matrix: list[float], epsilon: float) -> EdgeMatchStructure:
    """Compute edge match structure with epsilon accuracy."""
    vertex_match = construct_vertex_match(graph, distance_matrix, epsilon / 2)

    throughs: list[Through] = []
    for e1, e2 in graph.edges:
        # Only add one direction as trough object
        if e1 < e2:
            a = graph.vertices[e1]
            b = graph.vertices[e2]
            throughs.append(Through(a, b, e1, e2, epsilon))

    through_structure = calculate_through_structure(throughs)

    cuboids = [structure.bounding_cube.cuboid for structure in through_structure]
    tree = create_segmenttree_nd(cast(list[HyperInterval], cuboids), 3)
    return EdgeMatchStructure(vertex_match, through_structure, tree, cuboids, epsilon)

def edge_rectangle_intersect(a: Point, b: Point, rect: Rectangle) -> tuple[Point, Point] | None:
    """Intersect line segment ab to a rectangle. If the intersection is empty, then None is returned."""
    events: list[float] = [0, 1]
    if not epsilon_equal(a.x, b.x):
        events.append((rect.x.left - a.x) / (b.x - a.x))
        events.append((rect.x.right - a.x) / (b.x - a.x))
    if not epsilon_equal(a.y, b.y):
        events.append((rect.y.left - a.y) / (b.y - a.y))
        events.append((rect.y.right - a.y) / (b.y - a.y))

    unit_interval = Interval(0, 1)
    filtered_events = [event for event in events if
                       in_interval(event, unit_interval) and
                       in_rectangle(lerp_point(a, b, event), rect)]

    if len(filtered_events) >= 2:
        return (lerp_point(a, b, min(filtered_events)), lerp_point(a, b, max(filtered_events)))
    else:
        return None

def query_edge_match(graph: Graph, graph_neighbour: GraphNeighbour, edge_match: EdgeMatchStructure, square_origin: Point, square_radius: float) -> tuple[list[GraphMatch], EdgeMatchQueryInfo]:
    """Query edge match structure with epsilon accuracy. This is similar to T3 in the paper."""
    T2 = query_vertex_match(graph, graph_neighbour, edge_match.vertex_match, square_origin, square_radius)

    query_point = Point3D(square_origin.x, square_origin.y, square_radius)

    smallest_cube_size = INFINITY
    smallest_cube_neighbours = None

    through_structure_indices = query_segmenttree_nd(edge_match.tree, query_point)

    smallest_cube_size = INFINITY
    smallest_cube_neighbours = None
    for through_index in through_structure_indices:
        _, cube, neighbours = edge_match.structure[through_index]
        if cube.side_length < smallest_cube_size:
            smallest_cube_size = cube.side_length
            smallest_cube_neighbours = neighbours

    if smallest_cube_neighbours is None:
        query_info = EdgeMatchQueryInfo(
            vertex_match_count = len(T2),
            edge_match_count = 0,
            total_match_count = len(T2),
            trough_count = 0,
            avg_trough_subdivision = 0
        )

        return T2, query_info

    intersecting_throughs: list[Through] = []
    for neighbour in smallest_cube_neighbours:
        if through_point_intersect(neighbour, query_point):
            intersecting_throughs.append(neighbour)

    if len(intersecting_throughs) == 0:
        query_info = EdgeMatchQueryInfo(
            vertex_match_count = len(T2),
            edge_match_count = 0,
            total_match_count = len(T2),
            trough_count = 0,
            avg_trough_subdivision = 0
        )

        return T2, query_info

    Sp = Rectangle(
        Interval(square_origin.x - 2 * square_radius, square_origin.x + 2 * square_radius),
        Interval(square_origin.y - 2 * square_radius, square_origin.y + 2 * square_radius)
    )

    T3: list[GraphMatch] = []

    sample_step = edge_match.epsilon * square_radius / 2
    chord_sample_count_total = 0

    for through in intersecting_throughs:
        assert through_point_intersect(through, query_point)
        assert epsilon_geq(point_point_distance(through.a, through.b), 0.5 * edge_match.epsilon * square_radius), "Invariant points in T3; see Lemma 25 in cpacked"

        clamped_edge = edge_rectangle_intersect(through.a, through.b, Sp)
        assert epsilon_leq(line_point_distance(through.a, through.b, square_origin), 4 * square_radius), "Invariant points in T3; see Lemma 25 in cpacked"

        if clamped_edge is not None:
            at, bt = clamped_edge
            atbt = point_point_distance(at, bt)
            samples: list[float] = [0, 1]

            chord_sample_count = math.floor(atbt / sample_step)
            chord_sample_count_total += chord_sample_count
            for index in range(1, chord_sample_count):
                samples.append(index * sample_step / atbt)

            for sample in samples:
                T3.append(GraphMatch(lerp_point(at, bt, sample), through.a_index, through.b_index))

    T23 = [*T2, *T3]

    query_info = EdgeMatchQueryInfo(
        vertex_match_count = len(T2),
        edge_match_count = len(T3),
        total_match_count = len(T23),
        trough_count = len(intersecting_throughs),
        avg_trough_subdivision = chord_sample_count_total/len(intersecting_throughs)
    )

    return T23, query_info
