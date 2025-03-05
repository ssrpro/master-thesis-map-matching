"""This modules has functions for reading input graph and making sure it is on general position."""

from __future__ import annotations

import heapq
import math
import random
import typing

from dataclasses import dataclass
from typing import NamedTuple, Any

from .utilities import INFINITY, lerp, epsilon_range, Point, point_point_distance, line_line_intersection, in_interval, Interval, lerp_point

import numpy as np
import scipy

class GraphEdge(NamedTuple):
    """Edge in a graph described as vertex indices."""
    left: int
    right: int

@dataclass(frozen=True)
class Graph:
    """Graph datastructure."""
    vertices: list[Point]
    edges: list[GraphEdge]

    @property
    def size(self) -> tuple[int, int]:
        """Return number of vertices and edges respectively of the graph."""
        return (len(self.vertices), len(self.edges))

class GraphNeighbour:
    """For each vertex index yields all neighbouring vertex indices."""
    neighbours: list[list[int]]

    def __init__(self, graph: Graph):
        """Computes for all vertices the neighbouring vertices connected by that vertex."""
        set_neighbours: list[set[int]] = []

        for _ in enumerate(graph.vertices):
            set_neighbours.append(set())

        for edge_left, edge_right in graph.edges:
            set_neighbours[edge_left].add(edge_right)
            set_neighbours[edge_right].add(edge_left)

        self.neighbours = []
        for set_neighbour in set_neighbours:
            self.neighbours.append(list(set_neighbour))

    def get_vertex_neighbours(self, vertex_index: int) -> list[int]:
        """Function that returns all neighbouring vertex indices of a given vertex index."""
        return self.neighbours[vertex_index]
    
class GraphDistanceMatrix:
    """Contains shortest path distances between any pair of vertices of a given graph."""

    distance_matrix: list[float]
    vertices_count: int

    def __init__(self, graph: Graph, neighbour: GraphNeighbour):
        """Calculate all vertex pair Euclidean distances using Dijkstra's algorithm."""

        self.distance_matrix = []
        self.vertices_count = len(graph.vertices)

        for _ in range(self.vertices_count * self.vertices_count):
            self.distance_matrix.append(INFINITY)

        for i_index in range(self.vertices_count):
            dijkstra_queue: list[tuple[float, int]] = []
            visited: set[int] = set()
            self.distance_matrix[i_index + i_index * self.vertices_count] = 0
            heapq.heappush(dijkstra_queue, (0.0, i_index))

            while len(dijkstra_queue) > 0:
                j_distance, j_index = heapq.heappop(dijkstra_queue)

                if j_index not in visited:
                    visited.add(j_index)

                    for k_index in neighbour.neighbours[j_index]:
                        new_distance = j_distance + point_point_distance(graph.vertices[j_index], graph.vertices[k_index])

                        if new_distance < self.distance_matrix[k_index + i_index * self.vertices_count]:
                            self.distance_matrix[k_index + i_index * self.vertices_count] = new_distance
                            heapq.heappush(dijkstra_queue, (new_distance, k_index))

    def vertex_distance(self, vertex_index_a: int, vertex_index_b: int) -> float:
        """Returns the distance of the shortest path from vertex_a to vertex_b and vice versa, since the graph is undirected."""
        return self.distance_matrix[vertex_index_a + self.vertices_count * vertex_index_b]

def create_points(n: int, d: float) -> list[Point]:
    points: list[Point] = []
    for _ in range(n):
        points.append(Point(random.uniform(-d, d), random.uniform(-d, d)))

    return points

def create_graph_from_points(points: list[Point], remove_count: int, remove_threshold_interval: tuple[int, int]) -> tuple[Graph, tuple[list[tuple[Point, Point, Point]], list[tuple[Point, Point]]]]:
    np_points = np.array(points)
    delaunay = scipy.spatial.Delaunay(np_points)

    edges: set[tuple[int, int]] = set()
    for simplex in delaunay.simplices:
        for i in range(3):
            edges.add(tuple(sorted((simplex[i].item(), simplex[(i + 1) % 3].item()))))

    original_edges_list: list[tuple[int, int]] = list(edges)
    edges_list: list[tuple[int, int]] = list(edges)
    debug_removed: list[tuple[Point, Point, Point]] = []
    debug_added: list[tuple[Point, Point]] = []

    for _ in range(remove_count):
        while True:
            i1 = random.randint(0, len(points) - 1)
            i2 = random.randint(0, len(points) - 1)

            if i1 == i2:
                continue

            ii1: tuple[int, int] = typing.cast(tuple[int, int], tuple(sorted([i1, i2])))

            p1 = points[i1]
            p2 = points[i2]

            removed_edges: list[int] = []

            duplicate = False
            for ii2 in range(len(edges_list)):
                if i1 == edges_list[ii2][0] and i2 == edges_list[ii2][1]:
                    duplicate = True
                    break

            if duplicate:
                continue

            intersection_count = 0
            for ii2 in range(len(original_edges_list)):
                i3 = original_edges_list[ii2][0]
                i4 = original_edges_list[ii2][1]

                if i3 in [i1, i2]:
                    continue

                if i4 in [i1, i2]:
                    continue

                p3 = points[i3]
                p4 = points[i4]

                t = line_line_intersection(p1, p2, p3, p4)
                if t is not None and in_interval(t[0], Interval(0, 1)) and in_interval(t[1], Interval(0, 1)):
                    intersection_count += 1

            if intersection_count < remove_threshold_interval[0] or intersection_count > remove_threshold_interval[1]:
                continue

            for ii2 in range(len(edges_list)):
                i3 = edges_list[ii2][0]
                i4 = edges_list[ii2][1]

                if i3 in [i1, i2]:
                    continue

                if i4 in [i1, i2]:
                    continue

                p3 = points[i3]
                p4 = points[i4]

                t = line_line_intersection(p1, p2, p3, p4)
                if t is not None and in_interval(t[0], Interval(0, 1)) and in_interval(t[1], Interval(0, 1)):
                    debug_removed.append((lerp_point(p3, p4, t[1]), p3, p4))
                    removed_edges.append(ii2)

            removed_edges = sorted(removed_edges, reverse=True)
            for removed_edge in removed_edges:
                edges_list.pop(removed_edge)

            edges_list.append(ii1)
            debug_added.append((p1, p2))

            break

    edges_set: set[tuple[int, int]] = set()

    for i1, i2 in edges_list:
        edges_set.add((i1, i2))
        edges_set.add((i2, i1))

    edges_result: list[GraphEdge] = [GraphEdge(i1, i2) for (i1, i2) in edges_set]
    return Graph(points, edges_result), (debug_removed, debug_added)

def create_graph_highway_from_graph(graph: Graph, d: float, highway_points: int, highway_thickness: float, exits: list[int]) -> Graph:
    assert highway_points % 4 == 0, "highway_points must be divisible by 4, to each highway lane has equal number of points."

    n = len(graph.vertices)
    output_vertices = list(graph.vertices)
    output_edges = list(graph.edges)

    single_highway_points = highway_points // 4
    q = highway_thickness / 10

    prev_ii = None
    for i in range(single_highway_points):
        t = lerp(i / (single_highway_points - 1), -d, d)
        ii = len(output_vertices)
        output_vertices.append(Point(-highway_thickness / 2 + q, t + 5 * q))
        if prev_ii is not None:
            output_edges.append(GraphEdge(prev_ii, ii))
            output_edges.append(GraphEdge(ii, prev_ii))
        prev_ii = ii

    prev_ii = None
    for i in range(single_highway_points):
        t = lerp(i / (single_highway_points - 1), -d, d)
        ii = len(output_vertices)
        output_vertices.append(Point(highway_thickness / 2 + 2 * q, t + 6 * q))
        if prev_ii is not None:
            output_edges.append(GraphEdge(prev_ii, ii))
            output_edges.append(GraphEdge(ii, prev_ii))
        prev_ii = ii

    prev_ii = None
    for i in range(single_highway_points):
        t = lerp(i / (single_highway_points - 1) + 3 * q, -d, d)
        ii = len(output_vertices)
        output_vertices.append(Point(t, -highway_thickness / 2 + 7 * q))
        if prev_ii is not None:
            output_edges.append(GraphEdge(prev_ii, ii))
            output_edges.append(GraphEdge(ii, prev_ii))
        prev_ii = ii

    prev_ii = None
    for i in range(single_highway_points):
        t = lerp(i / (single_highway_points - 1) + 4 * q, -d, d)
        ii = len(output_vertices)
        output_vertices.append(Point(t, highway_thickness / 2 + 8 * q))
        if prev_ii is not None:
            output_edges.append(GraphEdge(prev_ii, ii))
            output_edges.append(GraphEdge(ii, prev_ii))
        prev_ii = ii

    for exit_index in exits:
        highway_point = output_vertices[n + exit_index]

        min_i: int | None = None
        min_d: float | None = INFINITY
        for i, p in enumerate(graph.vertices):
            distance = point_point_distance(p, highway_point)
            if distance < min_d:
                min_d = distance
                min_i = i

        assert min_i is not None, "The list graph.vertices should have at least one point, making min_i not None."
        output_edges.append(GraphEdge(min_i, n + exit_index))
        output_edges.append(GraphEdge(n + exit_index, min_i))

    return Graph(output_vertices, output_edges)


def read_graph_from_disk(filename_vertices: str, filename_edges: str) -> Graph:
    """Read graph from specified filenames."""
    vertices: list[Point] = []
    edges: list[GraphEdge] = []

    with open(filename_vertices, "r", encoding="utf-8") as f:
        for line in f.readlines():
            x_str, y_str = line.strip().split(" ")
            x, y = (float(x_str), float(y_str))
            vertices.append(Point(x, y))

    with open(filename_edges, "r", encoding="utf-8") as f:
        for line in f.readlines():
            v1_str, v2_str, _ = line.strip().split(" ")
            v1, v2 = (int(v1_str), int(v2_str))
            edges.append(GraphEdge(v1, v2))

    return Graph(vertices, edges)

def sub_graph(graph: Graph, bounds: tuple[float, float, float, float]) -> Graph:
    """Creates a sub graph by filtering out points outside the specified bounds from [0, 1], which is relative to the
    minimum and maximum of the original vertex bounds."""
    xm = INFINITY
    xp = -INFINITY
    ym = INFINITY
    yp = -INFINITY

    for vertex in graph.vertices:
        xm = min(xm, vertex.x)
        xp = max(xp, vertex.x)
        ym = min(ym, vertex.y)
        yp = max(yp, vertex.y)

    bxm, bxp, bym, byp = bounds

    # Bounds should be within [0, 1] range.
    assert epsilon_range(bxm, 0.0, 1.0)
    assert epsilon_range(bxp, 0.0, 1.0)
    assert epsilon_range(bym, 0.0, 1.0)
    assert epsilon_range(byp, 0.0, 1.0)

    xm2 = lerp(bxm, xm, xp)
    xp2 = lerp(bxp, xm, xp)
    ym2 = lerp(bym, ym, yp)
    yp2 = lerp(byp, ym, yp)

    new_vertices: list[Point] = []
    new_edges: list[GraphEdge] = []

    # Given graph G = (G.V, G.E) and subgraph H = (H.V, H.E) of G. Since |H.V| <= |G.V| the vertex indices of H do not correspond to indices of G,
    # hence storing edges of G in H will be incorrect. Let I(S) denote index representation of list S. The translator T : I(G.V) -> I(H.V) allows
    # to transform vertex index from G to vertex index from H. So an edge of G.E named (i1, i2) can be translated to edge of H.E as (T(i1), T(i2)).
    new_vertices_translator: dict[int, int] = dict()

    for i, p in enumerate(graph.vertices):
        if epsilon_range(p.x, xm2, xp2) and epsilon_range(p.y, ym2, yp2):
            new_vertices_translator[i] = len(new_vertices)
            new_vertices.append(p)

    for i, j in graph.edges:
        p = graph.vertices[i]
        q = graph.vertices[j]
        if epsilon_range(p.x, xm2, xp2) and epsilon_range(p.y, ym2, yp2) and epsilon_range(q.x, xm2, xp2) and epsilon_range(q.y, ym2, yp2):
            new_edges.append(GraphEdge(new_vertices_translator[i], new_vertices_translator[j]))

    return Graph(new_vertices, new_edges)

def rescale_graph(graph: Graph, scale: float) -> Graph:
    """Translate and rescale graph vertices in the range [-scale, scale] while maintaining the aspect ratio."""
    xm = INFINITY
    xp = -INFINITY
    ym = INFINITY
    yp = -INFINITY

    for vertex in graph.vertices:
        xm = min(xm, vertex.x)
        xp = max(xp, vertex.x)
        ym = min(ym, vertex.y)
        yp = max(yp, vertex.y)

    diam = max(xp - xm, yp - ym)
    new_vertices: list[Point] = []

    xc = 0.5 * (xp + xm)
    yc = 0.5 * (yp + ym)

    for vertex in graph.vertices:
        new_x = (vertex.x - xc) / (0.5 * diam) * scale
        new_y = (vertex.y - yc) / (0.5 * diam) * scale
        new_vertices.append(Point(new_x, new_y))

    return Graph(new_vertices, edges=graph.edges)

def visualize_graph(ax: Any, graph: Graph):
    """Visualizes the graph using pyplot."""
    for i, j in graph.edges:
        ax.plot([graph.vertices[i].x, graph.vertices[j].x], [graph.vertices[i].y, graph.vertices[j].y], linewidth=1, color="k")

    ax.scatter([x for x, _ in graph.vertices], [y for _, y in graph.vertices], s=3.5, color="red")

def graph_closeness(graph: Graph):
    """Shows the minimum distance between pair of vertices in the graph, by max norm and min norm. This is used to check
    if the vertices inside the graph are in general position."""
    global_closeness = INFINITY
    component_closeness = INFINITY

    for i in range(len(graph.vertices)):
        for j in range(i + 1, len(graph.vertices)):
            p = graph.vertices[i]
            q = graph.vertices[j]

            dx = abs(p.x - q.x)
            dy = abs(p.y - q.y)
            dz = 0.5 * abs((p.x - p.y) - (q.x - q.y))

            global_closeness = min(max(dx, dy, dz), global_closeness)
            component_closeness = min(min(dx, dy, dz), component_closeness)

    return (global_closeness, component_closeness)

def graph_pertube(graph: Graph, radius: float, steps: int):
    """Add noise to the vertices of the input graph, so it is in general position, i.e. the graph closeness is high enough."""
    new_vertices: list[Point] = []

    for x, y in graph.vertices:
        dx = 0.0
        dy = 0.0

        for _ in range(steps):
            angle = random.uniform(0.0, 2.0 * math.pi)
            r = random.uniform(0.0, radius)
            dx += r * math.cos(angle)
            dy += r * math.sin(angle)

        new_vertices.append(Point(x + dx, y + dy))

    return Graph(new_vertices, graph.edges)

def graph_largest_component(graph: Graph, neighbour: GraphNeighbour) -> Graph:
    """Returns the largest connected component of a graph."""
    largest_component = []

    # Same kind of translator as new_vertices_translator in sub_graph(...)
    largest_translator: dict[int, int] = dict()
    visited: set[int] = set()

    for vertex_index, _ in enumerate(graph.vertices):
        if vertex_index not in visited:
            visited.add(vertex_index)

            component: list[int] = []
            translator: dict[int, int] = dict()

            translator[vertex_index] = len(component)
            component.append(vertex_index)

            queue: list[int] = []
            
            for neighbour_index in neighbour.neighbours[vertex_index]:
                if neighbour_index not in visited:
                    queue.append(neighbour_index)
                    visited.add(neighbour_index)

            while len(queue) > 0:
                new_index = queue.pop()

                translator[new_index] = len(component)
                component.append(new_index)

                for neighbour_index in neighbour.neighbours[new_index]:
                    if neighbour_index not in visited:
                        queue.append(neighbour_index)
                        visited.add(neighbour_index)

            if len(component) > len(largest_component):
                largest_component = component
                largest_translator = translator

    vertices_set = set(largest_component)

    new_vertices: list[Point] = []
    for vertex_index in largest_component:
        new_vertices.append(graph.vertices[vertex_index])

    new_edges: list[GraphEdge] = []
    for a, b in graph.edges:
        if a in vertices_set and b in vertices_set:
            new_edges.append(GraphEdge(largest_translator[a], largest_translator[b]))

    return Graph(new_vertices, new_edges)

class GraphPath:
    """Path of a graph described by vertex indices."""
    vertex_index: int
    prev: GraphPath | None

    def __init__(self, prev: GraphPath | None, vertex_index: int):
        self.prev = prev
        self.vertex_index = vertex_index

    @staticmethod
    def create_from_vertex_indices(vertex_indices: list[int]) -> GraphPath | None:
        """Creates a graph path from list of vertices."""
        prev_node = None
        for vertex_index in vertex_indices:
            prev_node = GraphPath(prev_node, vertex_index)
        return prev_node

    def get_vertices(self) -> list[int]:
        """Returns the forward iterator of the path from head to tail."""
        vertices: list[int] = []

        pointer = self
        while pointer is not None:
            vertices.append(pointer.vertex_index)
            pointer = pointer.prev

        return list(reversed(vertices))

    def __lt__(self, other: GraphPath):
        return self.vertex_index < other.vertex_index
