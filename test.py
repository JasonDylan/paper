import itertools


def find_not_zero_lv_index(arr:list):
    # 从后往前找,找到最后一个非0则返回
    # 全是0 返回-1
    if sum(arr) == 0:
        return -1
    len_arr = len(arr)
    for i in range(len_arr - 1, -1, -1):
        if arr[i] != 0:
        # 有非0 返回非0的下标
            return i
    # 如果没有找到满足条件的数，表示全非0 返回len-1
    return len_arr-1

def comb_allocations(decision:list[tuple], allocations:list[list[tuple]])->list[list[tuple]]:
    # allocations = []
    return allocations


def get_min_allocate_num(tasks:list, servers:list, levels:list) -> int:
    return  max(0, sum(tasks) - sum(servers[:-1]))


def allocate(tasks:list, servers:list, levels:list)->list[list[tuple]]:
    # 递归结束
    allocations = []
    if len(tasks)==1 and len(servers)==1:
        idx = 0
        allocate_num = servers[idx] # 理论当到最后一个元素的时候，前面已经分配完毕了
        allocations = [[(levels[idx], levels[idx], allocate_num)]]
    elif sum(servers) > sum(tasks):
        decisions = []
        last_task_lv_idx = find_not_zero_lv_index(tasks) # 获取未分配完的最大level的task的下标
        last_server_lv_idx = find_not_zero_lv_index(servers) # 获取未分配完的最大level的server的下标
        # 分配一个等级的servers
        # 分配方法
        # 1. 找到能分配的最小值和最大值：min n <- sum(tasks) <= sum(servers[:-1])+n and max 当前等级的task 总数
        min_allocate_num = get_min_allocate_num(tasks, servers, levels)
        max_allocate_num = tasks[-1]
        # 2. 获取所有分配min-max
        for allocate_num in range(min_allocate_num, max_allocate_num+1):
            a_decision = []
            # 任务等级，业务员等级，数量
            a_decision.append((levels[last_task_lv_idx], levels[last_server_lv_idx], allocate_num))
            a_tasks = tasks.copy()
            a_servers = servers.copy()
        # 3. 对所有分配组合进行分配, 更新servers/tasks, 默认当前servers 分配完毕,所以删除当前servers, 做进一步分配
            a_tasks[last_task_lv_idx]-=allocate_num
            a_servers[last_server_lv_idx]-=a_servers[last_server_lv_idx]
            allocations = allocate(a_tasks, a_servers, levels)
            new_allocations = comb_allocations(a_decision, allocations)
            a_allocations = []
        # 4. 对于所有的分配情况, 其生成的decision和allocate结果聚合
        
        # 5. 对上面的聚合结果进行聚合list[list[tuple]]+list[list[tuple]]
        # allocations = allocate(tasks, servers, levels)
        # new_allocations = comb_allocations(a_decision, allocations)
        pass
    elif sum(servers) <= sum(tasks):
        last_task_lv_idx = find_not_zero_lv_index(tasks) # 获取未分配完的最大level的task的下标
        last_server_lv_idx = find_not_zero_lv_index(servers) # 获取未分配完的最大level的server的下标
        # 1. 当lvidx不同的时候,也就是意味着last_task_lv_idx > last_server_lv_idx当前level的server能分配的task很多, 
        #    生成分配所有servers[last_idx]到tasks[-1]~tasks[-last_task_lv_idx]的,参考之前写的方法。
        # 2. 相同时, 表示当前level的server能分配的task只有一个, 则完全分配最后一个
        # 1.2的都作为decision
        # 3. 更新tasks, servers(不需要删除 task )后, 调用allocations = allocate(tasks, servers, levels)
        # 4. new_allocations = comb_allocations(a_decision, allocations)
        pass

    return allocations
decisions = allocate([0, 0, 1, 3, 2], [2, 1, 1, 2, 1], [1, 2, 3, 4, 5])
print(decisions)

