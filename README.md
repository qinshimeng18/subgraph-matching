# 感悟
- 多进程计算中要巧用共享变量和queue
> 尽量避免用sharedmemory 写数据需要进行lock，降低性能