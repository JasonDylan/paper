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
    print(f"{decision=} {allocations=}")
    # 将decision 放入每个allcations
    if allocations:
        new_allocations = []
        for a_list in  allocations:
            a_list.extend(decision)
            new_allocations.append(a_list)
    else:
        new_allocations = [decision]
    return new_allocations


def get_min_allocate_num(tasks:list, servers:list, levels:list, last_task_lv_idx:int, last_server_lv_idx:int) -> int:
    left_servers_more_than_tasks_num = sum(tasks[:last_task_lv_idx+1]) - sum(servers[:last_server_lv_idx])
    return max(0, min(tasks[last_task_lv_idx], left_servers_more_than_tasks_num))


def allocate(tasks:list, servers:list, levels:list)->list[list[tuple]]:
    last_task_lv_idx = find_not_zero_lv_index(tasks) # 获取未分配完的最大level的task的下标
    last_server_lv_idx = find_not_zero_lv_index(servers) # 获取未分配完的最大level的server的下标
    print(f"allocate {tasks=} {servers=} {levels=} {last_task_lv_idx=} {last_server_lv_idx=}")
    # 递归结束
    all_allocations = []
    if last_task_lv_idx==-1 or -1==last_server_lv_idx:
        print("case 1")
        idx = 0
        allocate_num = servers[idx] # 理论当到最后一个元素的时候，前面已经分配完毕了
        all_allocations = []

    elif sum(servers) > sum(tasks):
        print("case 2")
        # 分配一个等级的servers
        # 1. 找到能分配的最小值和最大值：min n <- sum(tasks) <= sum(servers[:-1])+n and max 当前等级的task 总数
        min_allocate_num = get_min_allocate_num(tasks, servers, levels, last_task_lv_idx, last_server_lv_idx)
        max_allocate_num = tasks[last_task_lv_idx]
        print(min_allocate_num, max_allocate_num)
        # 2. 获取所有分配min-max
        for allocate_num in range(min_allocate_num, max_allocate_num+1):
            # if allocate_num:
            a_decision = []
            # 任务等级，业务员等级，数量
            if allocate_num>0:
                a_decision.append((levels[last_task_lv_idx], levels[last_server_lv_idx], allocate_num))
            a_tasks = tasks.copy()
            a_servers = servers.copy()
        # 3. 对所有分配组合进行分配, 更新servers/tasks, 默认当前servers 分配完毕,所以删除当前servers, 做进一步分配
            a_tasks[last_task_lv_idx] -= allocate_num
            can_cut_servers = False
            if last_server_lv_idx == last_task_lv_idx:
                if a_tasks[last_task_lv_idx]==0:
                    pass 

            if can_cut_servers:
                a_servers[last_server_lv_idx] -= a_servers[last_server_lv_idx]
            else:
                a_servers[last_server_lv_idx] -= allocate_num
            a_allocations = allocate(a_tasks, a_servers, levels)
        # 4. 对于所有的分配情况, 其生成的decision和allocate结果聚合
            new_allocations = comb_allocations(a_decision, a_allocations)
        # 5. 对上面的聚合结果进行聚合list[list[tuple]]+list[list[tuple]]
            all_allocations.extend(new_allocations)
    elif sum(servers) <= sum(tasks):
        print("case 3")
        # 1. 当lvidx不同的时候,也就是意味着last_task_lv_idx > last_server_lv_idx当前level的server能分配的task很多, 
        #    生成分配所有servers[last_idx]到tasks[-1]~tasks[-last_task_lv_idx]的,参考之前写的方法。
        if last_task_lv_idx > last_server_lv_idx:
            # 生成所有可分配组合
            allocate_lv_idx_range_list = list(range(last_server_lv_idx, last_task_lv_idx+1))
            allocate_ranges = [range(0, min(tasks[idx], servers[last_server_lv_idx])+1) for idx in allocate_lv_idx_range_list]
            allocate_combinations = list(itertools.product(*allocate_ranges))
            # 分配的server的lv
            to_allocated_lv = levels[last_server_lv_idx]
            to_allocated_server_num = servers[last_server_lv_idx]
            # 把comb的idx 转为 level的idx的list， 输入idx，返回level的idx ，用这个idx去levels 里取可以取搭配level的值
            print("start for loop")
            for comb in allocate_combinations:
                if sum(comb) == to_allocated_server_num:
                    print(f"avaliable comb {comb}")
                    a_decision = []
                    a_servers = servers.copy()
                    a_tasks = tasks.copy()
                    # 将生成的分配组合进行分配
                    for a_lv_idx, servered_num in zip(allocate_lv_idx_range_list, comb):
                        if servered_num:
                            a_lv = levels[a_lv_idx]
                            # 任务等级，业务员等级，数量
                            a_decision.append((a_lv, to_allocated_lv, servered_num))
                            # 对a_servers/a_tasks 进行分配更新
                            a_servers[last_server_lv_idx] -= servered_num
                            a_tasks[a_lv_idx] -= servered_num       
                    a_allocations = allocate(a_tasks, a_servers, levels)
                # 4. 对于所有的分配情况, 其生成的decision和allocate结果聚合
                    new_allocations = comb_allocations(a_decision, a_allocations)
                    all_allocations.extend(new_allocations)

        elif last_server_lv_idx == last_server_lv_idx:
        # 2. 相同时, 表示当前level的server能分配的task只有一个, 则完全分配最后一个
            a_servers = servers.copy()
            a_tasks = tasks.copy()
            allocate_num = a_servers[last_server_lv_idx]
            allocate_lv = levels[last_server_lv_idx]
            a_decision = []
            a_decision.append((allocate_lv, allocate_lv, allocate_num))
            a_servers[last_server_lv_idx] -= allocate_num
            a_tasks[last_task_lv_idx] -= allocate_num
            a_allocations = allocate(a_tasks, a_servers, levels)
            all_allocations = comb_allocations(a_decision, a_allocations)
        else:
            raise Exception(f"Error {last_server_lv_idx} {last_task_lv_idx} ")
        # 1.2的都作为decision
        # 3. 更新tasks, servers(不需要删除 task )后, 调用allocations = allocate(tasks, servers, levels)
        # 4. new_allocations = comb_allocations(a_decision, allocations)

    return all_allocations


def allocate_from_top(tasks:list, servers:list, levels:list)->list[list[tuple]]:
       
    pass


tasks = [2, 2, 0, 0, 0]
servers = [2, 6, 0, 0, 0]
decisions = allocate(tasks, servers, [1, 2, 3, 4, 5])
print(decisions)
print(decisions)

