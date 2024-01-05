import itertools


def find_not_zero_level_index(arr:list):
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

def allocate(tasks:list, servers:list, levels:list, decision:list[tuple]= []):
    # 传入当前state:(task/server/分配情况) ， 返回对于对于当前state的分配情况

    # 分配所有情况
    decisions = []
    last_task_lv_idx = find_not_zero_level_index(tasks) # 获取未分配完的最大level的task的下标
    last_server_lv_idx = find_not_zero_level_index(servers) # 获取未分配完的最大level的server的下标
    if last_task_lv_idx < last_server_lv_idx: # 当server level 多于 task 表示
        last_server_lv_idx = last_task_lv_idx
    if last_server_lv_idx==-1:
        decisions.append(decision)
        print(f"生成一个 {decisions=}")
        return decisions
    
    if last_task_lv_idx == last_server_lv_idx:
        # 分配最后一个非0 server给小于等于其level的task
        lv_idx = last_task_lv_idx # last_server_lv_idx 一样
        last_level = levels[lv_idx]

        server_num = servers[lv_idx] # 为什么全部都分配了？
        task_num = tasks[lv_idx]

        allocate_num = min(server_num, task_num)
        # 任务等级，业务员等级，数量
        decision.append((last_level, last_level, allocate_num))
        
        servers[lv_idx] = server_num - allocate_num
        tasks[lv_idx] = task_num - allocate_num

        a_decisions = allocate(tasks, servers, levels, decision) # 分配剩余
        decisions = decisions + a_decisions
    elif last_task_lv_idx > last_server_lv_idx:
        # 当最低等级servers剩余，将该servers分到小于等于该等级的task里
        allocate_ranges = [range(0, min(tasks[idx], servers[last_server_lv_idx])+1) for idx in range(last_server_lv_idx, last_task_lv_idx+1)]
        allocate_combinations = list(itertools.product(*allocate_ranges))
        to_allocated_level = levels[last_server_lv_idx]
        to_allocated_server_num = servers[last_server_lv_idx]
        # 筛选小于to_allocated_server_num条件的随机组合
        allocate_combinations_avalible = []
        idx_2_lv_idx = list(range(last_server_lv_idx, last_task_lv_idx+1))
        # 把comb的idx 转为 level的idx的list， 输入idx，返回level的idx ，用这个idx去levels 里取可以取搭配level的值
        filtered_combinations = [t for t in allocate_combinations if sum(t) == 2]
        print(filtered_combinations)
        for comb in filtered_combinations:
            print(f"avaliable comb {comb}")
            a_decision = decision.copy()
            a_servers = servers.copy()
            a_tasks = tasks.copy()
            for a_lv_idx, a_comb_item in zip(idx_2_lv_idx, comb):
                if a_comb_item:
                    a_level = levels[a_lv_idx]
                    servered_num = a_comb_item
                    # 任务等级，业务员等级，数量
                    a_decision.append((a_level, to_allocated_level, servered_num))
                    task_num = tasks[a_lv_idx]
                    remain_task_num = task_num - servered_num
                    remain_server_num = a_servers[last_server_lv_idx] - servered_num
                    a_servers[last_server_lv_idx] = remain_server_num
                    a_tasks[a_lv_idx] = remain_task_num
            new_decision = allocate(a_tasks, a_servers, levels, a_decision)
            decisions += new_decision
                
        # 对每个comb都做递归的添加，先求得分配之后的情况，对每种情况做分配
    elif last_task_lv_idx < last_server_lv_idx:
        
        print(f"error in {tasks=} {servers=} {levels=}")

    return decisions

decisions = allocate([0, 0, 1, 3, 2], [2, 1, 1, 2, 1], [1, 2, 3, 4, 5])
print(decisions)


# def allocate2(task_list: list[int], service_list: list[int], task_out: int, task_more: int, level_now: int, level_max: int)->list[list[tuple]]:
#     if level_now==level_max:
#         return 
#     result = []
#     task_out+=task_list[level_now]

    
#     for task_more_now in range(1+max(task_more, task_out-service_list[level_now])):
#         for
#             # 当前任务分配
            
#             # 上一级任务分配
#             result_tmp = allocate2(task_list, service_list, task_out, task_more_now, level_now-1, level_max)

#             # 任务分配组合



# def disatch_task()