import itertools

def find_not_zero_level_index(arr):
    if sum(arr) == 0:
        return -1
    
    len_arr = len(arr)
    for i in range(len_arr - 1, -1,-1):
        if arr[i] != 0:
            return i

    # 如果没有找到满足条件的数，则返回-1或其他合适的值
    return len_arr-1


def allocate(tasks, servers, levels, decision = []):
    # 当时最后一次分配
    # 当
    
    decisions = []
    last_task_level_idx = find_not_zero_level_index(tasks)
    last_server_level_idx = find_not_zero_level_index(servers)
    if last_server_level_idx==-1:
        decisions.append(decision)
        print(f"生成一个 decisions {decisions}")
        return False, decisions
    if last_task_level_idx == last_server_level_idx:
        # 分配最后一个非0 server给小于等于其level的task
        last_level = levels[last_server_level_idx]
        server_num = servers[last_server_level_idx]
        task_num = tasks[last_server_level_idx]
        # 任务等级，业务员等级，数量
        decision.append((last_level, last_level, server_num))
        remain_task_num = task_num - server_num
        servers[last_server_level_idx] = 0
        tasks[last_server_level_idx] = remain_task_num
        is_done, a_decisions = allocate(tasks, servers, levels, decision)
        print(type(a_decisions),a_decisions)
        decisions = decisions +a_decisions
    elif last_task_level_idx > last_server_level_idx:
        # 对一个servers 分到小于等于level的task里面
        # 随机
        # 筛选

        allocate_ranges = [range(0, min(tasks[idx]+1, servers[last_server_level_idx]+1)) for idx in range(last_server_level_idx, last_task_level_idx+1)]
        allocate_combinations = list(itertools.product(*allocate_ranges))
        to_allocated_level = levels[last_server_level_idx]
        to_allocated_server_num = servers[last_server_level_idx]
        # 筛选小于to_allocated_server_num条件的随机组合
        allocate_combinations_avalible = []
        idx_2_level_idx = list(range(last_server_level_idx, last_task_level_idx+1)) # 把comb的idx 转为 level的idx的list， 输入idx，返回level的idx ，用这个idx去levels 里取可以取搭配level的值
        
        for comb in allocate_combinations:
            print(f"comb {comb}")
            if sum(comb) == to_allocated_server_num:
                
                print(f"avaliable comb {comb}")
                a_decision = decision.copy()
                a_servers = servers.copy()
                a_tasks = tasks.copy()
                for a_level_idx, a_comb_item in zip(idx_2_level_idx, comb):
                    if a_comb_item:
                        a_level = levels[a_level_idx]
                        servered_num = a_comb_item
                        # 任务等级，业务员等级，数量
                        a_decision.append((a_level, to_allocated_level, servered_num))
                        task_num = tasks[a_level_idx]
                        remain_task_num = task_num - servered_num
                        remain_server_num = a_servers[last_server_level_idx] - servered_num
                        a_servers[last_server_level_idx] = remain_server_num
                        a_tasks[a_level_idx] = remain_task_num
                is_done, new_decision = allocate(a_tasks, a_servers, levels, a_decision)
                decisions += new_decision
                # if not is_done:
                #     break
        # 对每个comb都做递归的添加，先求得分配之后的情况，对每种情况做分配
    else:
        print(f"error in tasks, servers, levels{tasks, servers, levels}")

    return True, decisions


tasks =  [2, 2, 11]
servers = [3, 4, 1]
levels =  [1, 2, 3]
is_done, decisions = allocate(tasks, servers, levels, decision = [])
print(decisions)
for decision in decisions:
    print(decision)

# 先排除当天放假的员工

# 根据levels， servers，tasks区分组别

# 组别内allocate

# 对每一个decision，对于不同地方和不同的人，进行排列组合分配