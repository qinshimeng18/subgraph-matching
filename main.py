#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
from graph import Graph
from querygraph import QueryGraph
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import deque, OrderedDict
import sys
import timeit
from pprint import pprint
from loadxml import loadxml
# reload(sys)
# sys.setdefaultencoding('utf-8')


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
    inner, outer = splitvertices(matching)
    pos = nx.spring_layout(G)
    labels = {}
    for i in inner:
        labels[i] = 'P=' + i[:10]
    for i in outer:
        labels[i] = 'A=' + i[:10]
    # authour
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
    nx.draw_networkx_labels(G, pos, labels=labels, alpha=0.4, font_size=10)
    # Highlight matching edges
    nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.3)
    nx.draw_networkx_edges(G, pos,
                           edgelist=list(matching),
                           width=0.6, alpha=0.5,  edge_color='black')


def edges_vertices_from_json(path):
    """
    return
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


def networkx_graph(vertices, edges, name='default'):
    """
    利用networkx绘图
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    degree = nx.degree_histogram(G)
    # 返回图中所有节点的度分布序列
    # x = range(len(degree))  # 生成x轴序列，从1到最大度
    # y = [z / float(sum(degree)) for z in degree]
    # print x
    # 将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
    # plt.loglog(x, y, color="blue", linewidth=2)
    # # 在双对数坐标轴上绘制度分布曲线
    # plt.show()  # 显示图表
    # print G.edges()
    formatMatching(G, edges)
    plt.savefig(name + '.png')
    plt.show()
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
        dblp = json.load(json_file)['dblp']['article']
    # ablp=[{author:--,title:---,month:---,volume:--,year:--},{},{}]
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


def set_query_graph(query_graph):
    """
    vertex id is number, category and label matters
    Returns:
        [query graph] -- [description]
    """

    query_graphs = json.load(open('query_graph.json', 'r'))
    vertices = query_graphs[query_graph]['vertices']
    edges_weight = query_graphs[query_graph]['edges_weight']
    q = QueryGraph(vertices, edges_weight, u0='1', is_directed=False)
    return q


def print_graph(g):
    print '边的数量： ', len(g.get_edges())
    print '点的数量： ', len(g.get_vertices())
    # print 'edges:   ',  dict(g.get_edges())
    # print 'edges:   ',  g.get_edges()


def filter1_in_category_set(q):
    return q.category_list


def out_gephi_csv(edges):
    with open('edges.csv', 'w') as f:
        for i in edges:
            f.write('\'' + str(i[0]) + '\'' + ',' +
                    '\'' + str(i[1]) + '\'' + '\n')


def children_lost(q, u, level):
    """[summary]q,array, level
    lost score will >= real lost score for the edges may match in other match
    """
    # lost_score = 0
    lost_edges = set()
    neighbours = filter_by_level(q, q.neighbours(u), level)
    for child in neighbours:
        lost_edges.add((u, child))
        # lost_score += q.adjacency_dict[u][child]
        lost_score_child, lost_edges = children_lost(q, u, level + 1)
        lost_edges |= lost_edges
        # lost_score += lost_score_child
    return lost_edges


def judge(u, i, j, q, g):
    if q.get_vertex_category(i) == g.get_vertex_category(j):
        return q.adjacency_dict[u][i]
    else:
        return 0


def setLevel(q, root):
    level = 0
    queue = []
    queue_back = []
    queue.append(root)
    while 1:
        if len(queue) == 0 and len(queue_back) == 0:
            break
        while len(queue) > 0:
            node = queue.pop()
            q.set_vertex_level(node, level)
            children = q.neighbours(node)
            for i in children:
                if not q.has_vertex_level(i) and i not in queue_back:
                    queue_back.append(i)
        level += 1
        if len(queue) == 0 and len(queue_back) == 0:
            break
        while len(queue_back) > 0:
            node = queue_back.pop()
            q.set_vertex_level(node, level)
            children = q.neighbours(node)
            for i in children:
                if not q.has_vertex_level(i) and i not in queue:
                    queue.append(i)
        level += 1


def filter_by_level(q, array, level):
    """[children down]
    make sure node wont go back to a circle
    """
    result = []
    for i in array:
        # print 'node',i,' level',q.get_vertex_level(i),' >? ',level
        if(q.get_vertex_level(i) > level):
            result.append(i)
    return result


def judge_array(array):
    for i in array:
        if i != 0:
            return False
    return True


def poss_combination(a, b, temp, result):
    # print temp
    length = len(temp)
    # print length
    if judge_array(a) or judge_array(b):
        result.append(temp)
        return result
    for i in range(len(a)):
        for j in range(length, len(b)):
            if a[i] != 0 and b[j] != 0:
                temp_ = copy.deepcopy(temp)
                temp_.append((a[i], b[j]))
                a_i = a[i]
                b_j = b[j]
                a[i] = 0
                b[j] = 0
                # print "array", a, b
                poss_combination(a, b, temp_, result)
                a[i] = a_i
                b[j] = b_j
    return result


def get_combinations(a, b, temp):
    if len(a) < len(b):
        a, b = b, a
    return poss_combination(a, b, temp, [])

def get_lost_score(q,lost_edges):
    score = 0
    for i in lost_edges:
        score+=q.adjacency_dict[i[0]][i[1]]
    return score
def isSame(u, v, q, g, level=0):
    """[summary]

    """
    # print 'u:',u
    # print 'v:',v
    print '\n'
    children_u = q.neighbours(u)
    children_v = g.neighbours(v)
    children_u = filter_by_level(q, children_u, level)  # not go back
    print u, '`s children_u un-filter', children_u, '  current level is; ', level
    # print 'current level:',level
    # print u,'`s children_u',children_u
    level += 1
    matchArray = []
    match_score = 0
    lost_edges = set()
    m_u_list = set()
    l_u_list = set()
    # matched.append((u,v))
    # o(n^3)  n*(n-1)*(n-2)....
    combinations = get_combinations(children_u, children_v, [])
    """ [ [(Um,Vn),(U,V)] , [(),()] ]"""

    # print '---combinations--:',combinations
    if combinations and combinations[0]:
        u_index, v_index = (0, 1) if combinations[0][0][0].isalnum() else (1, 0)
        for combination in combinations:
        """[  ]"""
            simi_score = 0
            di_simi_score = 0
            di_simi_edges=set()

            line_matched = True
            lost_u_vertices = set()
            match_u_vertices = set()
            # 得到u_children匹配的结点对（即匹配出发边）、损失边及其损失点不在匹配边的权重之和
            for item in combination:
                judge_score = judge(u, item[u_index], item[v_index], q, g)
                if judge_score:
                    
                    match_u_vertices.add(item[u_index])
                    simi_score += judge_score
                else:
                    di_simi_edges |= children_lost(q, item[u_index], level+1)

            lost_u_vertices = set(children_u) - match_u_vertices
            for node in lost_u_vertices:
                di_simi_edges |= children_lost(q, node, level+1)
            di_simi_score = get_lost_score(q,di_simi_edges)


            if match_score == 0 or match_score < judge_score:
                match_score = simi_score
                matchArray = combination
                lost_score = di_simi_score
                lost_edges = di_simi_edges
                # ç = judge()
            #     if judge_score:
            #         simi_score+=judge_score
            #     else:
            #         line_matched = False
            #         break
            # if line_matched:
            #     if match_score == 0 or match_score < judge_score :
            #         match_score += simi_score
            #         matchArray = combination
    print 'matchArray:', matchArray
    print 'matchscroe:', match_score
    print 'di_simi_score',di_simi_score
    print 'di_simi_edges',di_simi_edges
    for i in matchArray:
        m_u_list.add((u, i[u_index]))
        # l_u_list
        # visited.extend([i[0], i[1]])
        # print i,u_index,v_index
        # print 'in-level--:',level
        # print u,'m_u_list',m_u_list
        print 'do matching :', i
        m, l, m_u, l_u = isSame(i[u_index], i[v_index], q, g, level)
        match_score += m
        lost_score += l
        # print 'm_u_list,',m_u_list
        m_u_list = m_u_list | m_u
        # l_u_list = l_u_list|l+u
        # break

    return match_score, lost_score, m_u_list, l_u_list

    #------------
    """
    for i in children_u:
        i_matched = False  # True if i has matched
        # print 'i: ',i
        visited_edges.append((u,i))
        # lost_edges.append((u,i))
        for j in children_v:
            # print 'j: ',j
            # print 'matched:',matched
            # print 'visited:',visited
            # j who haven't matched
            if (v,j) not in visited_edges and (i,j) not in matched and j not in visited:
                judge_score = judge(u, i, j, q, g)
                if(judge_score):
                    i_matched = True
                    # matched.append(j)
                    match_score += judge_score
                    # print 'matched score: ',match_score
                    # print 'lost score :   ',lost_score
                    # print i,j
                    matchArray.append((i, j))
                    visited_e.append([(u,i),(v,j)])
                    break
        if not i_matched:
            # print 'not match'
            if q.has_children(i, u):
                # lost_score += children_lost(q, i,u, visited_edges,lost_edges)
                pass
            else:
                lost_score += q.adjacency_dict[u][i]
    # ---------------
    # for _ in matchArray:
    #     if _ in visited_edges:
    #         print "wrong: repeat edges"
    #     else:
    #     visited_edges.append(_)
    for i in matchArray:
        visited.extend([i[0], i[1]])
        m, l = isSame(i[0], i[1], q, g, visited, matched,visited_e,level+1)
        match_score += m
        lost_score += l
    return match_score, lost_score
    """


def main():
    # xml to json ; then json to graph
    # xml_in = 'small.xml'
    # json_out = 'small.json'
    # loadxml(xml_in,json_out)

    # create Data graph and Query graph
    k = 10
    graph_path = 'small.json'
    query_graph = "graph2"
    graph_focus = []
    lost_edges = []
    edges, vertices = label_edges_vertices_from_json(graph_path)
    g = Graph(vertices, edges, False)  # 边的数量会多一半
    q = set_query_graph(query_graph)
    print_graph(g)
    u0 = '1'  # the first node in query graph
    setLevel(q, u0)
    print "====setlevel====="

    pprint(q.get_vertices())
    # 得到graph中所有的focus点（用category做比较）
    remove_num = 0
    for vertex in g.get_vertices():
        # print vertex,q.get_u0_category()
        if g.get_vertex_category(vertex) == q.get_u0_category():
            graph_focus.append(vertex)
        # delete vertices whose label are not in the query
        if g.get_vertex_category(vertex) not in filter1_in_category_set(q):
            g.remove_vertex(vertex)
            remove_num += 1
    # graph_focus == vertices
    #
    focus_score = OrderedDict()
    matched_graphs = OrderedDict()
    visited_vs = OrderedDict()  # del
    visited_es = OrderedDict()  # del
    count = 0

    for v0 in graph_focus:
        matched = []
        visited = [u0, v0]
        visited_e = []
        level = 0
        focus_score[v0] = isSame(u0, v0, q, g, level=level)
        matched_graphs[v0] = matched
        visited_vs[v0] = visited
        visited_es[v0] = visited_e
        # print '-------------'
        break
        # if count>1:
        #     break
        # else:
        #     count+=1
        #     print '======================================================'

    top_k = sorted(focus_score.iteritems(),
                   key=lambda d: d[1][0], reverse=True)[:k]
    for _ in top_k:
        print "graph:\n", _
        print "- matched_graphs: \n", matched_graphs[_[0]]
        # print "- visited_vs: \n",visited_vs[_[0]]
        # print "- visited_es: \n",visited_es[_[0]]
        print '---------------'
    # gg=networkx_graph(g.get_vertices(),edges,'g')
if __name__ == '__main__':
    print(timeit.timeit("main()", setup="from __main__ import main", number=1))
