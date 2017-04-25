#!/usr/bin/python
## -*- coding:utf-8 -*-
"""
graph.py

Implemented in lectures in CMPUT274 taught by Michael Bowman, Walter Bischov,
Leah Hackman & Zack friggstadt. Updated by Parash Rahman & Jacob Denson.
"""
class Graph:
    """
    Implements a graph class with standard features expected in a graph, like
    searching, pathfinding, and finding the tangent line at a specific point ;)
    """

    def __init__(self, vertices = [], edges = [], is_directed = True):
        """
        Given a list of vertices and edges (a list of tuples of vertex pairs),
        this function creates a graph with a given set of vertices and edges.
        If no vertices and edges are specified, an empty graph is created. One
        can also specify whether the graph is directed or not, which by default
        is directed.

        >>> a = Graph()
        >>> a.adjacency_dict == {}
        True
        >>> b = Graph([1,2,3], [(1,2), (2,3)], True)
        >>> b.adjacency_dict
        {1: {2}, 2: {3}, 3: set()}
        """

        self.is_directed = is_directed
        self.vertices = vertices
        # 邻接表 
        self.adjacency_dict = {}

        for vertex in vertices.keys():
            self.add_vertex(vertex)

        for vertex_from, vertex_to in edges:
            self.add_edge(vertex_from, vertex_to)

    def get_vertex_category(self,vertex):
        return self.vertices[vertex]['category']
    def is_vertex(self, vertex):
        """
        Returns true if the given vertex is in the graph.

        >>> a = Graph([1,2])
        >>> a.is_vertex(1)
        True
        >>> a.is_vertex(3)
        False
        """

        return vertex in self.adjacency_dict.keys()

    def is_edge(self, vertex_from, vertex_to):
        """
        Returns true if the specified edge is in the graph.

        >>> a = Graph([1,2], [(1,2)])
        >>> a.is_edge(1,2)
        True
        >>> a.is_edge(2,1)
        False
        >>> a.is_edge(3,1)
        False
        """

        # If the first vertex isn't even in the graph, the edge itself can't be.
        if not self.is_vertex(vertex_from):
            return False

        return vertex_to in self.adjacency_dict[vertex_from]

    def get_vertices(self):
        """
        Returns a copy of the set of vertices in the graph.

        >>> a = Graph([1,2,3], [(1,2), (2,3)])
        >>> a.vertices() == {1,2,3}
        True
        """
        # for i in self.adjacency_dict.keys():
        #     yield i
        # return self.adjacency_dict.keys()
        return set(self.adjacency_dict.keys())
      
    def get_edges(self):
        """
        Returns a list of edges in the graph.

        >>> a = Graph([1,2,3], [(1,2), (2,3)])
        >>> a.edges()
        [(1, 2), (2, 3)]

        >>> b = Graph()
        >>> b.edges()
        []
        """

        edges = [(vertex_from, vertex_to)
            for vertex_from in self.adjacency_dict.keys()
            for vertex_to in self.adjacency_dict[vertex_from]]

        return edges

    def add_vertex(self, vertex):
        """rom].add(v
        Adds a vertex to the graph.

        >>> a = Graph()
        >>> a.add_vertex(1)
        >>> a.vertices()
        {1}
        >>> a.add_vertex(2)
        >>> a.vertices()
        {1, 2}
        """

        # Adding a vertex twice to the graph may have been made in error
        # made in error, we specifically let the user know.
        if self.is_vertex(vertex):
            raise ValueError("Vertex {} is already in graph".format(vertex))

        self.adjacency_dict[vertex] = set()

    def add_edge(self, vertex_from, vertex_to):
        """
        Given a tuple of 2 vertices, adds an edge between them. If the graph is
        undirected, both directions of edge are added. If the graph is directed,
        an edge from the first to the second is added. Two edges cannot exist
        between the same two nodes.

        >>> a = Graph([1,2])
        >>> a.adjacency_dict
        {1: set(), 2: set()}
        >>> a.add_edge(1,2)
        >>> a.adjacency_dict
        {1: {2}, 2: set()}

        >>> b = Graph([1,2], is_directed = False)
        >>> b.add_edge(1,2)
        >>> b.adjacency_dict
        {1: {2}, 2: {1}}
        """

        # if not self.is_vertex(vertex_from):
        #     raise ValueError("Vertex {} is not in graph".format(vertex_1))

        # if not self.is_vertex(vertex_to):
        #     raise ValueError("Vertex {} is not in graph".format(vertex_2))

        # if self.is_edge(vertex_from, vertex_to):
        #     raise ValueError("Edge {} already in graph".format(edge))

        if self.is_directed:
            self.adjacency_dict[vertex_from].add(vertex_to)

        else:
            self.adjacency_dict[vertex_from].add(vertex_to)
            self.adjacency_dict[vertex_to].add(vertex_from)

    def remove_vertex(self, vertex):
        """
        Removes a vertex from the graph, also removing all edges connected
        to the edge and from the edge in the graph.

        >>> a = Graph([1,2,3], [(1,2), (2,3),(3,1)], False)
        >>> a.remove_vertex(1)
        >>> a.adjacency_dict
        {2: {3}, 3: {2}}
        """

        for other_vertex in self.adjacency_dict[vertex]:
            self.adjacency_dict[other_vertex].discard(vertex)

        self.adjacency_dict.pop(vertex)

    def remove_edge(self, vertex_from, vertex_to):
        """
        Removes an edge from the graph

        >>> a = Graph([1,2], [(1,2),(2,1)])
        >>> a.remove_edge(1,2)
        >>> a.edges()
        [(2, 1)]

        >>> b = Graph([1,2], [(1,2)])
        >>> b.remove_edge(1,2)
        >>> b.edges()
        []
        """

        if not self.is_vertex(vertex_from):
            raise ValueError("Vertex {} is not in the graph".format(vertex_from))

        if not self.is_vertex(vertex_to):
            raise ValueError("Vertex {} is not in the graph".format(vertex_to))

        if not self.is_edge(vertex_from, vertex_to):
            raise ValueError("""Edge from {} to {} does
                                not exist""".format(vertex_from, vertex_to))

        if self.is_directed:
            self.adjacency_dict[vertex_from].remove(vertex_to)

        else:
            self.adjacency_dict[vertex_from].remove(vertex_to)
            self.adjacency_dict[vertex_to].remove(vertex_from)

    def neighbours(self, vertex,visited=[]):
        """
        Given a vertex, returns a list of vertices reachable from that vertex.

        >>> g = Graph([1,2,3], [(1,2), (1,3)])
        >>> g.neighbours(1)
        [2, 3]
        """

        if vertex not in self.adjacency_dict.keys():
            raise ValueError("Vertex {} is not in graph".format(vertex))

        return [neighbour for neighbour in self.adjacency_dict[vertex] if neighbour not in visited] 












