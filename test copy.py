servers = [3, 1, 5]  # A
tasks = [2, 1, 3]    # B
levels = [2, 3, 4]
allocate_template = [0] * sum(tasks)  # C
all_allocations = [allocate_template.copy()]  # D

for idx in range(len(tasks)):
    a_allocate_template = allocate_template.copy()
    if idx == 1:
        a_allocate_template[0:tasks[idx]] = [levels[idx]] * tasks[idx]
    else:
        for k in range(idx + 1):
            if k < idx:
                a_allocate_template[sum(tasks[:k+1]):sum(tasks[:k+2])] = [levels[k]] * tasks[idx]
            else:
                remaining_servers = servers[idx] - tasks[idx]
                if remaining_servers > 0:
                    a_allocate_template[sum(tasks[:k+1]):] = [levels[k]] * tasks[idx]
                    servers[idx] = remaining_servers
                else:
                    break
    all_allocations.append(a_allocate_template)

print(all_allocations)

import itertools

def allocate(servers:list, tasks:list, levels:list):
    # for i in range(task_len):
    # levels
    # allocate_lv_idx_range_list = list(range(last_server_lv_idx, last_task_lv_idx+1))

    task_len = sum(tasks)
    allocate_ranges = [levels for idx in range(task_len)]
    allocate_combinations = list(itertools.product(*allocate_ranges))
    
    selected_combinations = []
    for comb in allocate_combinations:
        #满足两个条件的则放入
        # 对comb 相同等级分配的比如 / 同时一个tasks的分组里，
        if comb 