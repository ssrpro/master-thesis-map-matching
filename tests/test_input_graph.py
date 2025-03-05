import math
import random

from mapmatch.input_graph import Graph, GraphEdge, sub_graph, GraphPath
from mapmatch.utilities import Point

def create_cyclic_graph(point_count: int, origin: Point, radius: float) -> Graph:
    vertices: list[Point] = []
    edges: list[GraphEdge] = []

    for i in range(point_count):
        angle = i / point_count * math.pi * 2.0
        vertices.append(Point(radius * math.cos(angle) + origin.x, radius * math.sin(angle) + origin.y))

    for i in range(point_count):
        edges.append(GraphEdge(i, (i + 1) % point_count))

    return Graph(vertices, edges)

def test_sub_graph():
    graph = create_cyclic_graph(12, Point(0.5, 0.5), 2)
    sub = sub_graph(graph, (0, 0.55, 0, 0.55))

    assert len(sub.vertices) == 4
    assert len(sub.edges) == 3

def test_path():
    indices = [random.randint(0, 16) for _ in range(32)]

    path = None
    for index in indices:
        path = GraphPath(path, index)

    assert path is not None
    assert path.get_vertices() == indices
    
    pointer = GraphPath.create_from_vertex_indices(indices)
    while pointer is not None or path is not None:
        assert pointer is not None and path is not None
        assert pointer.vertex_index == path.vertex_index
        pointer = pointer.prev
        path = path.prev
