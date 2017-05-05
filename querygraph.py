#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
graph.py

Implemented in lectures in CMPUT274 taught by Michael Bowman, Walter Bischov,
Leah Hackman & Zack friggstadt. Updated by Parash Rahman & Jacob Denson.
"""


class QueryGraph:
    """
    Implements a graph class with standard features expected in a graph, like
    searching, pathfinding, and finding the tangent line at a specific point ;)
    """

    def __init__(self, vertices=[], edges_weight=[],u0='1', is_directed=True):

        self.is_directed = is_directed
        self.vertices = vertices
        # 邻接表
        self.adjacency_dict = edges_weight
        # u0
        self.u0 = u0
        self.category_list = [info['category'] for id,info in  self.vertices.items()]

    def get_vertex_category(self, vertex):
        return self.vertices[vertex]['category']

    def is_vertex(self, vertex):

        return vertex in self.adjacency_dict.keys()

    def is_edge(self, vertex_from, vertex_to):

        # If the first vertex isn't even in the graph, the edge itself can't
        # be.
        if not self.is_vertex(vertex_from):
            return False

        return vertex_to in self.adjacency_dict[vertex_from]
    def get_u0_category(self):
        return self.get_vertex_category(self.u0)
    def get_u0(self):
        return self.u0
    def get_vertices(self):

        return set(self.adjacency_dict.keys())

    def get_edges(self):
        """[summary]

            edges_weight = {
        '1': {
            2: 0.3
        },
        '2': {
            '3': 0.8,
            '1': 0.3
        },
        '3': {
            '1': 0.8
        }
    }

        Returns:
            [list] -- [(edgesFrom,edgesTo)]
        """
        edges = [(vertex_from, vertex_to)
                 for vertex_from in self.adjacency_dict.keys()
                 for vertex_to in self.adjacency_dict[vertex_from]]

        return edges

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
            raise ValueError(
                "Vertex {} is not in the graph".format(vertex_from))

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
    def has_children(self,i,u):
        return True if self.neighbours(i,visited = [u]) else False
    def neighbours(self, vertex,visited = []):
        """
        Given a vertex, returns a list of vertices reachable from that vertex.

        >>> g = Graph([1,2,3], [(1,2), (1,3)])
        >>> g.neighbours(1)
        [2, 3]
        """

        if vertex not in self.adjacency_dict.keys():
            raise ValueError("Vertex {} is not in graph".format(vertex))
        return [neighbour for neighbour in self.adjacency_dict[vertex].keys() if neighbour not in visited]
    def set_vertex_level(self,u,level):
        self.vertices[u]['level'] = level
        # print 'level: ',self.vertices[u]['level']
    def has_vertex_level(self,u):
        # print self.vertices
        # print self.vertices[u],"has level?: ",self.vertices[u].has_key('level')
        if 'level' in self.vertices[u]:
            return True
        else:
            return False
    def get_vertex_level(self,u):
        return self.vertices[u]['level']
    def get_vertices(self):
        return self.vertices