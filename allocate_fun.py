

import itertools
from collections import Counter

# class 

def is_comb_valid(comb, levels, servers):
    count = Counter(comb)
    for level in levels:
        if level not in count or count[level] > servers[levels.index(level)]:
            return False
    return True


def is_comb_valid2(comb, levels, tasks):
    last_task_num = 0
    len_tasks = len(tasks)
    for idx, task_num in enumerate(tasks):
        # print(comb[last_task_num:last_task_num+task_num], "\r")
        for allocate_level in comb[last_task_num:last_task_num+task_num]:
            if idx+1 < len_tasks and  allocate_level in levels[idx+1:]:
                return False
        last_task_num = last_task_num+task_num
    return True


def encode_comb(comb):
    return '-'.join(str(num) for num in comb)


def decode_comb(comb_str):
    return tuple(int(num) for num in comb_str.split('-'))


def encode_key(servers:list, tasks:list, levels:list):
    return f"{servers}-{tasks}-{levels}"


def decode_key(encoded_key):
    servers_str, tasks_str, levels_str = encoded_key.split('-')
    servers = list(eval(servers_str))
    tasks = list(eval(tasks_str))
    levels = list(eval(levels_str))
    return servers, tasks, levels

#TODO 打表，加快生成速度
all_server_task_level_2_comb = []
def allocate(servers:list, tasks:list, levels:list) -> list[tuple]:
    print(f"{servers=}\n{tasks=}\n{levels=}")
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
        if is_comb_valid(comb, levels, servers) and is_comb_valid2(comb, levels, tasks):
            # print(f"{comb=}")
            selected_combinations.append(comb)
    return selected_combinations


avaliable_combinations = allocate(servers=[3,1,5], tasks=[2,1,3], levels=[2,3,4])