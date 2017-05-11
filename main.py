#!/usr/bin/python
# -*- coding:utf-8 -*-
from multiprocessing import Pool
import json
from graph import Graph
from querygraph import QueryGraph
import networkx as nx
import matplotlib.pyplot as plt
import copy
from collections import deque, OrderedDict
import sys
import os
import getopt
import timeit
from pprint import pprint
from loadxml import loadxml
import time
# reload(sys)
# sys.setdefaultencoding('utf-8')
u0_infuence = 0


def printOrWrite(content, writeToFileFlag, f):
    if writeToFileFlag:
        f.write(str(content))
        f.write("\n")
    else:
        print str(content)


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
        [query graph] -- 
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
    array - neighbours
    level +  1
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


def get_lost_score(q, lost_edges):
    score = 0
    for i in lost_edges:
        score += q.adjacency_dict[i[0]][i[1]]
    return score


def get_u_edges(q):
    # for i in q.
    pass


def filter1_in_category_set(q, g):
    for vertex in g.get_vertices():
        if g.get_vertex_category(vertex) not in q.category_list:
            g.remove_vertex(vertex)


def filter2_lost_gt_minVk(lost_children, lost_score, q, u, level, min_Vk_Score):
    dissimilarity = 0
    for lost_node in lost_children:
        dissimilarity += q.adjacency_dict[u][lost_node]
        for child_child in filter_by_level(q, q.neighbours(lost_node), level):
            dissimilarity += q.adjacency_dict[lost_node][child_child]
    if (lost_score + dissimilarity) >= (u0_infuence - min_Vk_Score):
        return -1
    else:
        return dissimilarity


def filter_uv_children(u, v, q, g, level):
    global matched_tracks
    # print 'matched_tracks:'
    # pprint(matched_tracks)
    # for node in g.neighbours(v):
    #     if (node, v) in matched_tracks['matched_v_e']:
    #         print '(node, v) in matched_tracks[\'matched_v_e\']',str(node),str(v)
    #     if (v, node) in matched_tracks['matched_v_e']:
    #         print '(v, node) in matched_tracks[\'matched_v_e\']',str(v),str(node)
    children_v = [node for node in g.neighbours(v) if ((
        v, node) not in matched_tracks['matched_v_e'] and (node, v) not in matched_tracks['matched_v_e'])]  # g不走重复的路
    children_u = filter_by_level(q, q.neighbours(u), level)  # not go back
    children_u = [node for node in children_u if (
        u, node) not in matched_tracks['matched_u_e']]  # q不走重复的路
    return children_u, children_v


def isSame(u, v, q, g, level=0, lost_score=0, min_Vk_Score=-1, filterFlag=1):
    # print '\n'##pp1
    global u0_infuence
    global matched_tracks
    children_u, children_v = filter_uv_children(u, v, q, g, level)
    if len(children_u) == 0 or len(children_v) == 0:
        return {'node':[],'links':[],'q': set(), 'g': set(), 'focus': '', 'match_score': 0}
    # print 'u:',u,' \t children:',children_u
    # print 'v:',v,'\t children:',children_v
    level += 1
    matchArray = []
    lost_children = set()
    m_u_list = set()
    ret = {'node':[],'links':[],'q': set(), 'g': set(), 'focus': '', 'match_score': 0}
    category = 0 if q.vertices[u]['category']=='paper' else 1
    ret['node'].append((v,category,v))
    # category = 0 if q.vertices[i[0]]['category']=='paper' else 1 
    # ret['node'].append({'id':i[1],'category':category,'name':i[1]})
    # o(n^3)  n*(n-1)*(n-2)....
    combinations = get_combinations(children_u, children_v, [])
    """ [ [(Um,Vn),(U,V)] , [(),()] ]"""
    # 得到u_children匹配的结点对
    if combinations and combinations[0]:
        u_index, v_index = (0, 1) if combinations[0][
            0][0].isdigit() else (1, 0)
        # 遍历所有情况，求当前simi_score最大的一组children匹配方式
        for combination in combinations:
            simi_score = 0
            simi_edges = []
            for item in combination:
                # print 'combination',combination
                judge_score = judge(u, item[u_index], item[v_index], q, g)
                if judge_score:
                    simi_edges.append((item[u_index], item[v_index]))
                    simi_score += judge_score
            # 更新simi_score最大的组合
            if ret['match_score'] == 0 or ret['match_score'] < simi_score:
                ret['match_score'] = simi_score
                matchArray = simi_edges
        matcth_children = set([mline[0] for mline in matchArray])
        lost_children = set(children_u) - matcth_children

    filter_result = 0
    if filterFlag:
        filter_result = filter2_lost_gt_minVk(
            lost_children, lost_score, q, u, level, min_Vk_Score)
    if filter_result == -1:
        return {'node':[],'links':[],'q': set(), 'g': set(), 'focus': '', 'match_score': -1}
    else:
        lost_score += filter_result
    # print 'matchArray',matchArray,'\n'
    matched_tracks['matched_v_e'].extend([(v, uv[1]) for uv in matchArray])
    for i in matchArray:
        matched_tracks['matched_u_e'].append((u, i[0]))
        ret['q'].add((u, i[0]))
        ret['g'].add((v, i[1]))
        ret['links'].append({'id':len(ret['links']),'source':v,'target':i[1]})
        category = 0 if q.vertices[u]['category']=='paper' else 1
        ret['node'].append((v,category,v))
        category = 0 if q.vertices[i[0]]['category']=='paper' else 1 
        ret['node'].append((i[1],category,i[1]))
        
        # print ret
        ret_temp = isSame(i[0], i[1], q, g,
                          level, lost_score, min_Vk_Score, filterFlag)
        if ret_temp['match_score'] == -1:  # m = -1 means v is not in M(G,Q,Vk)
            return {'node':[],'links':[],'q': set(), 'g': set(), 'focus': '', 'match_score': -1}
        ret['match_score'] += ret_temp['match_score']
        ret['q'] = ret['q'] | ret_temp['q']
        ret['g'] = ret['g'] | ret_temp['g']
        ret['node'].extend(ret_temp['node'])
        ret['links'].extend(ret_temp['links'])
       
        # break

    return ret


def get_u0_infuence(q):
    return (reduce((lambda x, y: x + y), [weight for value in q.adjacency_dict.values() for weight in value.values()])) / float(2)


def get_graph_focus(q, g):
    graph_focus = []
    for vertex in g.get_vertices():
        # print vertex,q.get_u0_category()
        uo_category = q.get_u0_category()
        if g.get_vertex_category(vertex) == uo_category:
            graph_focus.append(vertex)
        # delete vertices whose label are not in the query

    return graph_focus
def return_Q(q):
    nodes = []
    for id,d in q.vertices.items():
        if d['category'] == 'paper':
            category = 0
        else:
            category = 1
        nodes.append({'id':id,'category':category,'name':id})
    edges = []
    index = 0
    for key,valuses in q.adjacency_dict.items():
        for v,weight in valuses.items():
            edges.append({'id':index,'source':key,'target':v})
            index += 1
    return {'nodes':nodes,'edges':edges}
def main(graph_path='data.json', query_graph='graph1', k=1, filterFlag=1, commend=1):
    # xml to json ; then json to graph
    # xml_in = 'small.xml'
    # json_out = 'small.json'
    # loadxml(xml_in,json_out)
    graph_focus = []
    min_Vk_Score = -1
    if commend == 1:  # 命令行运行
        k = 1  # 自定义
        graph_path = 'data.json'
        query_graph = "graph3"
        filterFlag = True
        opts, args = getopt.getopt(sys.argv[1:], 'hg:q:k:f:')
        for opt, value in opts:  # 或者从命令行获取参数
            if opt == 'g':
                graph_path = value
            elif opt == 'q':
                query_graph = value
            elif opt == 'k':
                k = value
            elif opt == 'f':
                filterFlag = value
            elif opt == 'h':
                def usage():
                    print   """    
                            参数使用说明:  
                            -g data graph
                            -q query graph
                            -k top k 
                            -f filter 0/1
                            """
    elif commend == 0:  # 从web过来的命令 直接跳过。已赋值
        pass
    u0 = '1'  # the first node in query graph
    matched_graphs = []
    edges, vertices = label_edges_vertices_from_json(graph_path)
    g = Graph(vertices, edges, False)  # 边的数量会多一半
    q = set_query_graph(query_graph)
    start = time.time()
    setLevel(q, u0)
    if filterFlag:
        filter1_in_category_set(q, g)
    global u0_infuence
    u0_infuence = get_u0_infuence(q)
    graph_focus = get_graph_focus(q, g)
    # 得到graph中所有的focus点（用category做比较）
    count = 1
    global matched_tracks
    for v0 in graph_focus:
        # matched = []
        matched_tracks = {'matched_u': [], 'matched_v': [], 'matched_u_e': [
        ], 'matched_v_e': [], 'to_match_u': [], 'to_match_v': []}
        visited = [u0, v0]
        visited_e = []
        level = 0
        ret = isSame(
            u0, v0, q, g, level=level, lost_score=0, min_Vk_Score=min_Vk_Score, filterFlag=filterFlag)
        # -1 filter2 failed
        if ret['match_score'] == -1:
            continue
        # 更新相似topk的集合 & 得到最小匹配分
        ret['focus'] = v0
        if count < k:
            matched_graphs.append(ret)
        elif count == k:
            matched_graphs.append(ret)
            matched_graphs = sorted(matched_graphs,
                                    key=lambda d: d['match_score'], reverse=True)
            min_Vk_Score = matched_graphs[-1]['match_score']
        elif count > k:
            if ret['match_score'] > min_Vk_Score:
                matched_graphs.pop()
                matched_graphs.append(ret)
                matched_graphs = sorted(matched_graphs,
                                        key=lambda d: d['match_score'], reverse=True)
                min_Vk_Score = matched_graphs[-1]['match_score']
            else:   # not in the topk list
                pass
        count += 1
        # break
        
    for _ in matched_graphs:
        _['g'] = list(_['g'])
        _['q'] = list(_['q'])
        _['node'] = list(set(_['node']))
        temp=[]
        for i in _['node']:
            temp.append({'id':i[0],'category':i[1],'name':i[2]})
        _['node'] = temp
        index= 0
        for i in _['links']:
            i['id'] = index
            index +=1

    calc_time = (time.time() - start)
    writeToFileFlag = True
    f = open('mutil_proc.txt', 'a')
    printOrWrite('graph_path:' + graph_path + '\t' +
                 'query_graph:' + query_graph, writeToFileFlag, f)
    printOrWrite('calc_time:' + str(calc_time), writeToFileFlag, f)
    printOrWrite('边的数量：' + str(len(g.get_edges())), writeToFileFlag, f)
    printOrWrite('点的数量：' + str(len(g.get_vertices())), writeToFileFlag, f)
    printOrWrite('top-k:' + str(k), writeToFileFlag, f)
    printOrWrite('match_score: ' + '  '.join([str(i['match_score'])
                                              for i in matched_graphs]), writeToFileFlag, f)
    printOrWrite('graphs:' + str(matched_graphs), writeToFileFlag, f)
    f.close()
    graph_q = return_Q(q)
    return {'graph_path': graph_path,
            'query_graph': query_graph,
            'calc_time': calc_time,
            'top-k': k,
            'match_score': [i['match_score'] for i in matched_graphs],
            'matched_graphs': matched_graphs,
            'graph_q':graph_q
            }
    # printOrWrite(str(matched_graphs))
    # for _ in matched_graphs:
    #     printOrWrite(str(_))
    #     pprint(_)
    # print 'min_Vk_Score', min_Vk_Score
    # print get_u0_infuence(q)
if __name__ == '__main__':
    print(timeit.timeit("main(graph_path ='data.json',query_graph ='graph1',k =1,filterFlag = 1)",
                        setup="from __main__ import main", number=1))
    # f.close()
