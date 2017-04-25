#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
from graph import Graph
from querygraph import QueryGraph
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import deque
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def splitvertices(matching):
    """Split into vertices in and out of matching."""
    outer = set([])
    inner = set([])
    for (u, v) in matching:
        # if u in outer:
        #     outer.remove(u)
        # if v in outer:
        #     outer.remove(v)
        inner.add(u)
        outer.add(v)
    # print u, '#####', v
    return list(inner), list(outer)


def formatMatching(G, matching):
# G=nx.grid_2d_graph(2,1)  #4x4 grid
# plt.subplot(221)
# nx.draw(G,pos,font_size=8)
# plt.subplot(222)
# nx.draw(G,pos,node_color='k',node_size=0,with_labels=False)
# plt.subplot(223)
# nx.draw(G,pos,node_color='g',node_size=250,with_labels=False,width=6)
# #最后一幅子图转为有向图
# plt.subplot(224)
# H=G.to_directed()
# nx.draw(H,pos,node_color='b',node_size=20,with_labels=False)
# plt.savefig("four_grids.png")
# plt.show()
    inner, outer = splitvertices(matching)
    # print inner, outer
    pos = nx.spring_layout(G)
    # Blue = vertices _not_ in matching
    labels={}
    for i in inner:
        labels[i]='P='+i[:10]
    for i in outer:
        labels[i]='A='+i[:10]
    #authour
    nx.draw_networkx_nodes(G, pos,
                           nodelist=outer,
                           node_color='r',
                           node_size=50,
                           alpha=0.5, with_labels=True, font_size=8)

    # paper
    nx.draw_networkx_nodes(G, pos,
                           nodelist=inner,
                           node_color='g',
                           node_size=100,
                           alpha=0.7, with_labels=True, font_size=8)
    nx.draw_networkx_labels(G,pos,labels=labels,alpha=0.4,font_size=10)
    # Highlight matching edges
    nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.3)
    nx.draw_networkx_edges(G, pos,
                           edgelist=list(matching),
                           width=0.6, alpha=0.5,  edge_color='black')

    # Generate random matching, if not provided.
    # if not matching:
    #     matching = self.randomMatching()
    # # Convert to NetworkX
    # G = self.__toNetworkX()


def edges_vertices_from_json(path):
    """
    return:
        edges = [name,name]
        vertices = [(from,to)]
    """
    edges = []
    vertices = []
    with open(path, 'r') as json_file:
        data = json.load(json_file)

    dblp = json.loads(data)['dblp']['article']
    # 删去 不全的点
    dblp = [d for d in dblp if 'author' in d.keys()]
    # for paper in dblp:
    #   print
    title_author = dict([(d['title'], d['author'])
                         for d in dblp if 'author' in d.keys()])
    for vertex_from in title_author:
        if type(title_author[vertex_from]) != list:
            edges.append((vertex_from, title_author[vertex_from]))
        else:
            for vertex_to in title_author[vertex_from]:
                edges.append((vertex_from, vertex_to))
    vertices = set([i for i, j in edges])
    for e in edges:
        vertices = vertices | set(e)

    return edges, vertices


def networkx_graph(vertices,edges,name='default'):
    """
    利用networkx绘图
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    degree = nx.degree_histogram(G)
    # 返回图中所有节点的度分布序列
    x = range(len(degree))  # 生成x轴序列，从1到最大度
    y = [z / float(sum(degree)) for z in degree]
    # print x
    # 将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
    # plt.loglog(x, y, color="blue", linewidth=2)
    # # 在双对数坐标轴上绘制度分布曲线
    # plt.show()  # 显示图表
    # print G.edges()
    formatMatching(G, edges)
    plt.savefig(name+'.png')
    # plt.show()
    data = {}

    # pos = nx.spring_layout(G)          #定义一个布局，此处采用了spectral布局方式，后变还会介绍其它布局方式，注意图形上的区别
    # nx.draw(G,pos,with_labels=True,node_size = 100,font_size=8)  #绘制规则图的图形，with_labels决定节点是非带标签（编号），node_size是节点的直径
    # plt.show()  #显示图形  http://www.oschina.net/question/54100_77524

    # >>> a = Graph()
    #       >>> a.adjacency_dict == {}
    #       True
    #       >>> b = Graph([1,2,3], [(1,2), (2,3)], True)
    #       >>> b.adjacency_dict
    #       {1: {2}, 2: {3}, 3: set()}
    # g=Graph(vertices,edges,False)
    # g.adjacency_dict()
    # print g.adjacency_dict
    # gg=nx.Graph()
    # gg.add_edges_from(g.adjacency_dict)


def label_edges_vertices_from_json(path):
    """
    异构网络：人是种类category，教授、讲师和学生等是label
    return:
        vertices = {name_vertex:[name,category,weight ==1 ]}
        edges = [(name_from,name_to)]
    """

    edges = []
    vertices = {}
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    dblp = json.loads(data)['dblp']['article']
    # ablp=[{author:--,title:---,month:---,volume:--,year:--},{},{}]
    del data
    for i in dblp:
        if i.has_key('author') and i.has_key('title'):
            if type(i['title']) == dict:
                i['title'] = i['title']['#text']
            if type(i['author']) == list:
                vertices[i['title']] = {'category': 'paper', 'weight': 1}
                for j in i['author']:
                    vertices[j] = {'category': 'person', 'weight': 1}
                    edges.append((i['title'], j))
            else:
                vertices[i['author']] = {'category': 'person', 'weight': 1}
                vertices[i['title']] = {'category': 'paper', 'weight': 1}
                edges.append((i['title'], i['author']))
    del dblp
    return edges, vertices


def set_query_graph():
    """
    vertex id is number, category and label matters
    Returns:
        [query graph] -- [description]
    """
    vertices = {'1': {'category': 'paper', 'weight': 1}, '2': {
        'category': 'person', 'weight': 1}, '3': {'category': 'paper', 'weight': 1}, '4': {'category': 'paper', 'weight': 1}}
    edges_weight = {
        '1': {
            '2': 0.3
        },
        '2': {
            '3': 0.8,
            '1': 0.3,
            '4': 0.2
        },
        '3': {
            '2': 0.8
        },
        '4': {
            '2': 0.2
        }
    }
    q = QueryGraph(vertices, edges_weight, '1', False)
    return q


def print_graph(g):
    print '边的数量： ', len(g.get_edges())
    print '点的数量： ', len(g.get_vertices())
    # print 'edges:   ',  dict(g.get_edges())
    # print 'edges:   ',  g.get_edges()


def filter1_in_category_set(q):
    return q.category_list
def gephi(edges):
    with open('gephi.csv','w') as f:
        for i in edges:
            f.write('\''+str(i[0])+'\''+','+'\''+str(i[1])+'\''+'\n')

def main():
    path = 'json.json'
    graph_focus = []
    edges, vertices = label_edges_vertices_from_json(path)
    g = Graph(vertices, edges, False)  # 边的数量会多一半
    q = set_query_graph()
    # 得到graph中所有的focus点（用category做比较）
    remove_num = 0
    for vertex in g.get_vertices():
        # print vertex,q.get_u0_category()
        if g.get_vertex_category(vertex) == q.get_u0_category():
            graph_focus.append(vertex)
        # 对图进行修改
        if g.get_vertex_category(vertex) not in filter1_in_category_set(q):
            g.remove_vertex(vertex)
            remove_num += 1
    bfs = deque(graph_focus)
    foucus_num = 0
    for u in graph_focus:
        # print 'u^^^^^^^^^^^^^^^^^^^^^^^^',u
        foucus_num+=1
        visited = set()
        # 复制一个树
        G = copy.deepcopy(g)
        # 初始队列
        queue = deque()
        queue.append(u)
        visited.add(u)
        count = 0
        # print '开始queue: ',queue
        # print '开始visited: ',visited
        while queue:
            count+=1
            # print '-------开始nfs---处理V，将v的邻居加入queue和visited中----'
            # print 'step ',count
            v = queue.popleft()
            # print 'vertexv(当前循环节点): ',v
            neighbours = G.neighbours(v, visited)
            # print 'v的neighbours ',neighbours
            visited = visited.union(neighbours)
            queue.extend(neighbours)
            # print '开始queue: ',len(queue),queue
            # print '开始visited: ',len(visited),visited
            for neighbour in neighbours:
                neighbour_neighbours = G.neighbours(neighbour, visited)
                # print 'neighbour_neighbours',len(neighbour_neighbours),neighbour_neighbours
                visited = visited.union(neighbour_neighbours)
                if neighbour_neighbours:
                    queue.extend(neighbour_neighbours)
            # print '结束visited: ',len(visited),visited
            # print '结束queue: ',len(queue),queue
        print '--------循环结束-------'
        print 'count',count
        print 'visited: ',len(visited)
        break
        # u_now = bfs.popleft()
        # visited.append(u_now)
        # neighbours = G.neighbours(u_now)
        # delete_edges(neighbours,u_now)
        # for neighbour in neighbours:
        #     if g.get_vertex_category(neighbour) 交集 q.:
        #         sum()
        #     top[id:score]

        # bfs = []
        # bfs.append(u)
        # bfs+=G.neighbours(u)
    #不能同时打开会重叠的
    # qq=networkx_graph(q.get_vertices(),q.get_edges(),'q')
    # gg=networkx_graph(g.get_vertices(),edges,'g')
    # gephi graph_focus
    gephi(edges)
    print 'remove_num: ', remove_num
    print 'graph_focus: ', len(graph_focus)
    print 'foucus_num',foucus_num
    print_graph(g)
    print_graph(q)


if __name__ == '__main__':
    import timeit
    print(timeit.timeit("main()", setup="from __main__ import main", number=1))
    # networkx_graph(edges,vertices)
# if __name__ == '__main__':
#     main()
