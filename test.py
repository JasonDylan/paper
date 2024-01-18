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
    allocations = []
    a_allocation = []
    # 当前函数，默认每个task_num < server_num 
    # 从前往后遍历，获得task_num 和server_num
    # 第一个task_num 会被当前server_num 全部分配完成
    # 第二个task_num 需要分配时要，便利server_num是否分配给 task_num 的情况
    # 一个不分配，则第二个server_num 完全分配task_num, 剩余server_num 继续往下分配，而此时也有可能上上个level还有剩余，可以分配到这里
    # 这里应该是用之前用过的，随机分配方式，获得所有的组合，对于剩下的level[0:]
    # level 1 server 固定的 level 1 2 3 4
    #                            * (    ) = sum
    # 这里可以用矩阵来表示，当遍历到level n的时候，leveln的task 会被servers 必然分配完成，同时servers会有剩余
    # 而这部分剩余。排列组合分配到剩下的level n+1-m(m表示最后一个level)，也就构成了一个矩阵level_num * 排列组合数量
    # tasks - 这个矩阵，就能得到 剩余的task矩阵，在这个矩阵基础上，分配leveln+1
    # 到level n+1 的时候 leveln+1 
    last_level_left = servers[0]
    for task_num in tasks:
        allocate_num = task_num
        
    pass
def all_comb(a_tasks:list, a_servers:list, idx:int, allocate_num:int):
    a_servers_list = []
    a_tasks_list = []
    a_combs_list = []

    state_len = len(a_tasks)
    # 获取所有comb
    server_left = a_servers[idx]
    # allocate_lv_idx_range_list = list(range(idx+1, len_levels))
    allocate_lv_idx_range_list = list(range(state_len))
    allocate_ranges = [range(0, min(server_left, a_tasks[i])+1) for i in allocate_lv_idx_range_list]
    allocate_combinations = list(itertools.product(*allocate_ranges))
    # 在comb基础上更新后续的tasks,servers，然后传入进入下个
    for comb in allocate_combinations:
        if sum(comb) == server_left:
            # 此时分配就是 list(comb)[idx] = allcate_num
            a_comb = list(comb)

            a_tasks_after_comb = a_tasks.copy()
            a_servers_after_comb = a_servers.copy()

            a_tasks_after_comb =[x-y for x,y in zip(a_tasks_after_comb, a_comb)]
            a_servers_after_comb[idx] -= server_left

            a_comb[idx] += allocate_num # 这个必须得在a_tasks之后
            # a_comb_list.append(tuple(a_comb))
            print(f"对于分配{a_comb}后状态为{a_tasks_after_comb=} {a_servers_after_comb=}")
            # 此时 a_comb 是当前level分配的，一种分配方式

            a_servers_list.append(a_tasks_after_comb)
            a_tasks_list.append(a_servers_after_comb)

            a_combs_list.append(list(a_comb))
            # a_combs_list.append(list(tuple(a_comb)))
            # 保存每一个comb之后的状态和决策，后面循环/递归，需要根据目前状态继续分配下一个level，
    return a_servers_list, a_tasks_list, a_combs_list
def allocate_rec(tasks:list, servers:list, levels:list)->list[list[tuple]]:
    
    len_levels = len(levels)
    tasks_list = [tasks]
    servers_list = [servers]
    combs_list = [[]]
    all_combs_list = []
    for idx, level in enumerate(levels):
        print(f"当前分配状态{tasks=} {servers=}")
        # 求 每个level的servers的分配情况，分配了之后也是有状态的。
        # a_tasks_list = tasks_list.copy()
        # a_servers_list = servers_list.copy()
        a_comb_list = []
        # for a_tasks, a_servers, a_combs in zip(a_tasks_list, a_servers_list, a_combs_list):
        a_servers = servers.copy()
        a_tasks = tasks.copy()
        # 从前往后分配，先分配第一个
        allocate_num = a_tasks[idx]
        print(f"当前{level=} 待分配数量{a_servers[idx]=}")
        # 剩余server数据计算，计算之后分配给后level[idx+1:]
        server_left = a_servers[idx] - allocate_num
        # 分配后状态更新
        a_tasks[idx] = 0
        a_servers[idx] = server_left

        a_servers_list, a_tasks_list, a_combs_list = all_comb(a_tasks, a_servers, idx, allocate_num)
        
        # tasks_list = a_tasks_list
        # servers_list = a_servers_list
        # combs_list = a_combs_list
        # all_combs_list.append(combs_list)


    print(f"{tasks_list=}\n{servers_list=}\n{combs_list=}\n{all_combs_list=}")

tasks = [1, 2, 1, 0, 0]
servers = [3, 3, 0, 0, 0]
decisions = allocate_rec(tasks, servers, [1, 2, 3, 4, 5])
print(decisions)
print(decisions)

