# 感悟
- 多进程计算中要巧用共享变量和queue
> 尽量避免用sharedmemory 写数据需要进行lock，降低性能

## filters
1. 对Q中不包含的结点种类进行删除边操作
2. 当 当前V0的LostScore ≥ (u0_influence - Min_matched_score) 结束当前节点的计算  
> lost_score = lost_score + miss_children of v  
3. 父节点不在M(G, Q, k)中，则v也不在(废话)
## Q:
1. 环问题：setlevel
2. 重复走问题：
Q：level保证了不会成环，故不需要保存待访问边集合
G：匹配上的matchArray蕴含的边集合(包括待访问)和已经访问过的点matched_v_e 

## G:
1.
2.

##UV
matched_tracks =
 {'matched_u':[],
 'matched_v':[],
 'matched_u_e':[],
 'matched_v_e':[]
 ,'to_match_u':[],
 'to_match_v':[]}















