"""Helper functions to generate TikZ visualizations for map matching intermediate and end results."""

from .utilities import Point, point_point_distance, epsilon_equal, INFINITY, EPSILON
from .bar_tree import BarTreeNode, Region, BarTreeOneCutNode, BarTreeTwoCutNode, BarTreeLeafNode
from .input_graph import Graph, GraphNeighbour
from .graph_match import GraphMatch

def output_tikz(tikz_lines: list[str], name: str):
    with open(f"./output/{name}.tikz", "w") as file:
        file.write("\\begin{tikzpicture}\n")
        for tikz_line in tikz_lines:
            file.write(f"    {tikz_line}\n")
        file.write("\\end{tikzpicture}\n")

def visualize_graph(graph: Graph, name: str, size: float=2):
    output: list[str] = []

    for p in graph.vertices:
        output.append(f"\\fill ({p.x}, {p.y}) circle({size}pt);")

    for edge in graph.edges:
        p1 = graph.vertices[edge[0]]
        p2 = graph.vertices[edge[1]]
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    output_tikz(output, name)

def visualize_graph_debug(graph: Graph, points: tuple[list[tuple[Point, Point, Point]], list[tuple[Point, Point]]], name: str, size: float=2):
    output: list[str] = []

    for edge in graph.edges:
        p1 = graph.vertices[edge[0]]
        p2 = graph.vertices[edge[1]]
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    for _, p1, p2 in points[0]:
        output.append(f"\\draw[red, thick] ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    for p1, p2 in points[1]:
        output.append(f"\\draw[green, thick] ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    for px, _, _ in points[0]:
        output.append(f"\\draw[blue,fill=blue] ({px.x}, {px.y}) circle({size * 3/8}pt);")

    for p in graph.vertices:
        output.append(f"\\fill ({p.x}, {p.y}) circle({size}pt);")

    output_tikz(output, name)

def visualize_highway_debug(original_graph: Graph, graph: Graph, name: str, size: float=2):
    output: list[str] = []

    n = len(original_graph.vertices)
    for i, p in enumerate(graph.vertices):
        if i < n:
            output.append(f"\\fill ({p.x}, {p.y}) circle({size}pt);")
        else:
            output.append(f"\\fill[red] ({p.x}, {p.y}) circle({size}pt);")

    for i1, i2 in graph.edges:
        p1 = graph.vertices[i1]
        p2 = graph.vertices[i2]

        if i1 < n and i2 < n:
            output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")
        elif i1 >= n and i2 >= n:
            output.append(f"\\draw[red] ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")
        else:
            output.append(f"\\draw[green] ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    output_tikz(output, name)

def visualize_bar_tree(tree: BarTreeNode, name: str, points: list[Point], mode: int):
    """
    Visualizes BAR tree by drawing every cut.
    """

    def get_cut_line(region1: Region, region2: Region) -> tuple[Point, Point]:
        vertices1 = region1.get_vertices()
        vertices2 = region2.get_vertices()

        points: list[Point] = []
        for v1 in vertices1:
            for v2 in vertices2:
                if epsilon_equal(point_point_distance(v1, v2), 0):

                    duplicate = False
                    for v3 in points:
                        if epsilon_equal(point_point_distance(v1, v3), 0, 100 * EPSILON):
                            duplicate = True
                            break

                    if not duplicate:
                        points.append(v1)
                    break

        assert len(points) == 2
        return (points[0], points[1])

    def get_intersecting_regions() -> list[tuple[Point, Point]]:
        result: list[tuple[Point, Point]] = []

        queue: list[BarTreeNode] = []
        queue.append(tree)

        while len(queue) > 0:
            node = queue.pop()
            match node:
                case BarTreeOneCutNode() | BarTreeTwoCutNode():
                    if isinstance(node.children[0], (BarTreeOneCutNode, BarTreeTwoCutNode)) and isinstance(node.children[1], (BarTreeOneCutNode, BarTreeTwoCutNode)):
                        r1 = node.children[0].region
                        r2 = node.children[1].region
                        assert r1 is not None and r2 is not None

                        result.append(get_cut_line(r1, r2))

                        queue.append(node.children[0])
                        queue.append(node.children[1])
                case BarTreeLeafNode():
                    continue

        return result

    output: list[str] = []

    if mode in [0, 1]:
        for point in points:
            output.append(f"\\fill[blue2] ({point.x}, {point.y}) circle(1pt);")

    if mode in [1, 2]:
        root_vertices = tree.region.get_vertices()
        root_vertices = [*root_vertices, root_vertices[0]]
        for p1, p2 in zip(root_vertices[:-1], root_vertices[1:]):
            output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

        regions = get_intersecting_regions()
        for p1, p2 in regions:
            output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")
    output_tikz(output, name)

def visualize_bar_tree_query(tree: BarTreeNode, point: Point, name: str):
    """
    Visualizes BAR tree point query process using binary search, the numbers indicate the order
    in which the region is excluded in the binary search process.
    """

    def get_cut_line(region1: Region, region2: Region) -> tuple[Point, Point]:
        vertices1 = region1.get_vertices()
        vertices2 = region2.get_vertices()

        points: list[Point] = []
        for v1 in vertices1:
            for v2 in vertices2:
                if epsilon_equal(point_point_distance(v1, v2), 0):

                    duplicate = False
                    for v3 in points:
                        if epsilon_equal(point_point_distance(v1, v3), 0, 100 * EPSILON):
                            duplicate = True
                            break

                    if not duplicate:
                        points.append(v1)
                    break

        assert len(points) == 2
        return (points[0], points[1])
    
    def get_centroid(region: Region) -> Point:
        return Point(0.5 * (region.bounds.xm + region.bounds.xp), 0.5 * (region.bounds.ym + region.bounds.yp))

    def get_intersecting_regions() -> tuple[BarTreeLeafNode | None, list[tuple[Point, float, tuple[Point, Point]]]]:
        result: list[tuple[Point, float, tuple[Point, Point]]] = []
        node = tree
        leaf = None

        while True:
            match node:
                case BarTreeOneCutNode() | BarTreeTwoCutNode():
                    if isinstance(node.children[0], (BarTreeOneCutNode, BarTreeTwoCutNode)) and isinstance(node.children[1], (BarTreeOneCutNode, BarTreeTwoCutNode)):
                        r1: Region | None = None
                        r2: Region | None = None

                        match node.cut.axis:
                            case "x":
                                if point.x < node.cut.value:
                                    r1 = node.children[0].region
                                    r2 = node.children[1].region
                                    node = node.children[0]
                                else:
                                    r1 = node.children[1].region
                                    r2 = node.children[0].region
                                    node = node.children[1]
                            case "y":
                                if point.y < node.cut.value:
                                    r1 = node.children[0].region
                                    r2 = node.children[1].region
                                    node = node.children[0]
                                else:
                                    r1 = node.children[1].region
                                    r2 = node.children[0].region
                                    node = node.children[1]
                            case "z":
                                if point.z < node.cut.value:
                                    r1 = node.children[0].region
                                    r2 = node.children[1].region
                                    node = node.children[0]
                                else:
                                    r1 = node.children[1].region
                                    r2 = node.children[0].region
                                    node = node.children[1]
                        assert r1 is not None and r2 is not None
                        result.append((get_centroid(r2), min(r2.get_diam_i()) * 0.5, get_cut_line(r1, r2)))
                    else:
                        leaf = node.children[0]
                        break
                case BarTreeLeafNode():
                    leaf = node
                    break
        return (leaf, result)

    output: list[str] = []
    root_vertices = tree.region.get_vertices()
    root_vertices = [*root_vertices, root_vertices[0]]
    for p1, p2 in zip(root_vertices[:-1], root_vertices[1:]):
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    ss = INFINITY
    leaf, regions = get_intersecting_regions()
    for i, (c, s, (p1, p2)) in enumerate(regions):
        ss = min(ss, s)
        output.append(f"\\node[anchor=center] at ({c.x}, {c.y}) {'{'}")
        output.append(f"    \\resizebox{{{ss}cm}}{{!}}{ {i+1} }")
        output.append("};")

        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    if leaf is not None:
        centroid = get_centroid(leaf.region)
        size = min(leaf.region.get_diam_i())
        output.append(f"\\filldraw ({centroid.x},{centroid.y}) circle ({size * 0.25}cm);")

    output_tikz(output, name)

def visualize_sspd(graph: Graph, neighbour: GraphNeighbour, pp: int, qq: int, A: list[int], B: list[int], C: list[int], name: str, size: float=2):
    output: list[str] = []

    A0 = set(A)
    B0 = set(B)

    AA: set[int] = set()
    for aa0 in A:
        aa_queue = [aa0]
        
        while len(aa_queue) > 0:
            aa = aa_queue.pop()
            if aa not in AA:
                AA.add(aa)

                for aa_new in neighbour.get_vertex_neighbours(aa):
                    if aa_new not in AA and aa_new not in C:
                        aa_queue.append(aa_new)

    BB: set[int] = set()
    for bb0 in B:
        bb_queue = [bb0]
        
        while len(bb_queue) > 0:
            bb = bb_queue.pop()
            if bb not in BB:
                BB.add(bb)

                for bb_new in neighbour.get_vertex_neighbours(bb):
                    if bb_new not in BB and bb_new not in C:
                        bb_queue.append(bb_new)

    for edge in graph.edges:
        p1 = graph.vertices[edge[0]]
        p2 = graph.vertices[edge[1]]
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    for i, p in enumerate(graph.vertices):
        if i == pp:
            output.append(f"\\fill[blue2] ({p.x}, {p.y}) circle({size}pt);")
        elif i == qq:
            output.append(f"\\fill[red2] ({p.x}, {p.y}) circle({size}pt);")
        elif i in A0:
            output.append(f"\\fill[blue1] ({p.x}, {p.y}) circle({size}pt);")
        elif i in B0:
            output.append(f"\\fill[red1] ({p.x}, {p.y}) circle({size}pt);")
        elif i in C:            
            output.append(f"\\fill[green1] ({p.x}, {p.y}) circle({size}pt);")
        elif i in AA:
            output.append(f"\\fill[orange1] ({p.x}, {p.y}) circle({size}pt);")
        elif i in BB:
            output.append(f"\\fill[orange2] ({p.x}, {p.y}) circle({size}pt);")

    output_tikz(output, name)

def visualize_edge_match(graph: Graph, name: str, query_point: Point, edge_match_result: list[GraphMatch], size: float=2):
    output: list[str] = []

    points: list[Point] = []

    for point, _, _ in edge_match_result:
        points.append(point)

    output.append(f"\\fill[red] ({query_point.x}, {query_point.y}) circle({size * 1.2}pt);")

    for vertex in graph.vertices:
        output.append(f"\\fill[black] ({vertex.x}, {vertex.y}) circle({size}pt);")

    for point in points:
        output.append(f"\\draw ({point.x}, {point.y}) node[cross={size * 0.5}pt, line width={size * 0.125}pt, green]{{}};")

    for edge in graph.edges:
        p1 = graph.vertices[edge[0]]
        p2 = graph.vertices[edge[1]]
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    output_tikz(output, name)

def visualize_graph_trajectory(graph: Graph, name: str, trajectory: list[Point], size: float=2):
    output: list[str] = []

    for p in graph.vertices:
        output.append(f"\\fill ({p.x}, {p.y}) circle({size}pt);")

    for edge in graph.edges:
        p1 = graph.vertices[edge[0]]
        p2 = graph.vertices[edge[1]]
        output.append(f"\\draw ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    for p in trajectory:
        output.append(f"\\fill[green] ({p.x}, {p.y}) circle({0.5 * size}pt);")

    for p1, p2 in zip(trajectory[:-1], trajectory[1:]):
        output.append(f"\\draw[green] ({p1.x},{p1.y}) -- ({p2.x},{p2.y});")

    output_tikz(output, name)
