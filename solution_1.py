# %%
import pandas as pd
import numpy as np
# 读取 Excel 文件
import itertools
import sys

# %%
df = pd.read_excel('./data/中国各城市空间权重矩阵(1).xlsx', sheet_name='地理距离矩阵')
city_to_city_series = df[['cityeng', 'cityseries']].drop_duplicates().set_index('cityeng')['cityseries'].to_dict()

# %%
def change_df_city_name_2_idx(cities: pd.DataFrame)->(pd.DataFrame, dict):
    city_names = cities.columns
    city_num = len(city_names)
    city_name_nums = list(range(1, city_num+1))
    city_num_2_name = {key:value for key,value in zip(city_name_nums, city_names)}
    new_cities = cities.copy()
    new_cities.columns = new_cities.index = city_name_nums
    return new_cities, city_num_2_name


def generate_city(city_nums:int=26)->(pd.DataFrame, dict):
    df = pd.read_excel('./data/中国各城市空间权重矩阵(1).xlsx', sheet_name='地理距离矩阵')
    provinces = df.iloc[:, 0].values
    distance_matrix = df.iloc[:, 3:]
    city_index = 2
    distance_matrix.index = df.iloc[:, city_index].values
    distance_matrix.columns = df.iloc[:, city_index].values

    # 挑选出安徽、江苏、浙江和上海的省份及对应的矩阵数据
    selected_provinces = ['Anhui', 'Jiangsu', 'Zhejiang', 'Shanghai']
    selected_indices = [i for i, prov in enumerate(provinces) if prov in selected_provinces]
    selected_indices_str = df.values[selected_indices, city_index]
    print(f"{selected_indices_str=}")
    # 从城市列表中随机选择25个城市（不包括上海）
    selected_index = pd.Index(selected_indices_str)
    # 使用 loc 方法选择相应的行和列
    selected_data = distance_matrix.loc[selected_index, selected_index]
    random_cities = np.random.choice(selected_indices_str[1:], city_nums-1, replace=False)
    # 将上海添加到随机选择的城市列表中
    # City158 = shanghai
    selected_cities = ['City158'] + random_cities.tolist()
    selected_index = pd.Index(selected_cities)
    # 使用 loc 方法选择相应的行和列
    selected_data = selected_data.loc[selected_index, selected_index]
    city_to_city_series = df[['cityeng', 'cityseries']].drop_duplicates().set_index('cityeng')['cityseries'].to_dict()
    city_series_to_city = df[['cityeng', 'cityseries']].drop_duplicates().set_index('cityseries')['cityeng'].to_dict()
    city_series_columns = selected_data.columns
    city_columns = []
    for item in city_series_columns:
        city_columns.append(city_series_to_city[item])
    city = pd.DataFrame(selected_data.values, index=city_columns, columns=city_columns)

    new_cities, city_num_2_name = change_df_city_name_2_idx(cities=city)
    return new_cities, city_num_2_name


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
    # 分配所有情况
    
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
            # print(f"comb {comb}")
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

# 先判断是否可以分配，划分分配的区间，区间内任意分配，
def generate_joins(task_level_counts, server_level_counts):
    # 输入
    # task_level_counts pandas.core.series.Series(level_num,)
    # server_level_counts pandas.core.series.Series(level_num,)
    # server number tag task number
    # level 1 6 > 5
    # level 2 2 > 1
    # level 3 3 ≤ 6
    # level 4 2 > 1
    # level 5 4 ≤ 17
    # level 6 5 > 4
    # level 7 7 > 3
    # 需要根据不同等级任务和员工的等级做个计数判断，当连续的每个等级各自同时加起来大于现有的任务等级，表示这一组level之前可以任意分配，可以分配出不同的决策，
    # 比如上面的表里level1和level 2 server number >task number ，意味着，level 1可以分配给level2，也可以level2 都分配给level2，这是不同的决策空间，但是level 1 task也必须被做完。当level3 也考虑进去的时候，123相加就server数量就比task少了。这时候level4 就不能和123 一起考虑，但是123 之间可以一起决策，level3 全部分配到task，同时level 2 个也得分配到level3 或者level2
    # 输出 return joins = [[1,2,3],[4],[5]]

    # 输入校验：
    if isinstance(task_level_counts, pd.core.series.Series) \
        and isinstance(server_level_counts, pd.core.series.Series):
        if task_level_counts.shape[0] != server_level_counts.shape[0]:
            raise Exception("输入pandas.core.series.Series形状不一致")
    else:
        raise Exception(f"对象不是Pandas Series task_level_counts:{type(task_level_counts)}, server_level_counts:{type(server_level_counts)}")
    

    joins = []  # 存储决策空间的列表
    current_join = []  # 当前的决策空间划分
    task_level_num = task_level_counts.shape[0]      # 任务等级数量 
    task_join_count = 0
    server_join_count = 0

        # print("idx, task_count, server_count, task_join_count, server_join_count")
    for idx, (task_count, server_count) in enumerate(zip(task_level_counts, server_level_counts)):
        task_join_count += task_count
        server_join_count += server_count
        # print(idx, task_count, server_count, task_join_count, server_join_count)
        level = idx+1
        if server_join_count <= task_join_count:
            # 当任务数量大于员工数量时，将当前决策空间划分添加到列表中
            
            if current_join:
                current_join.append(level)
                joins.append(current_join)
            else:
                joins.append([level])
                
            current_join = []  # 重置当前决策空间划分
            task_join_count = 0
            server_join_count = 0
        else:
            # 当任务数量小于等于员工数量时，将当前等级添加到当前决策空间划分中
            
            current_join.append(level)
    
    if current_join:
        joins.append(current_join)  # 将最后一个决策空间划分添加到列表中
    
    return joins


def get_combinations(server_list, cities, current_combination, all_combinations=[], min_sum=float('inf'), min_combination=None):
    global a_city
    if len(server_list) == 0:
        all_combinations.append(current_combination)
        current_sum = sum([a_city[server][city] for server,city in current_combination])
        if current_sum < min_sum:
            min_sum = current_sum
            min_combination = current_combination
        return min_sum, min_combination

    for i, city in enumerate(cities):
        remaining_cities = cities[:i] + cities[i+1:]
        new_current_combination = current_combination + [(server_list[0], city)]
        min_sum, min_combination = get_combinations(server_list[1:], remaining_cities, new_current_combination, all_combinations, min_sum, min_combination)
    
    return min_sum, min_combination

# 任务等级，业务员等级，数量
# task_level, server_level, allocate_num 
# 同时要递归地进入
# 对于一个decisions 内的单个allocation，进行分配
# 输入为指定task_level的城市位置，有指定server_level对应的server位置, 和分配数量，
# 然后得到城市的位置range和server的位置range， 随机匹配allocate_num的个数量，约束是如果以及分配完的目的地不应该再次分配
def allocate_servers_2_cities_for_a_decision(allocate_tuple, initial_state_df, servers_remain_df):
    global revenue_for_level, a_city
    # print()
    task_level, server_level, allocate_num = allocate_tuple
    revenue = revenue_for_level[task_level-1] * allocate_num
    cost = 0
    # print("task_level, server_level, allocate_num ", task_level, server_level, allocate_num )
    # servers_remain_df
    # 对于一个decisions 内的单个allocation，进行分配
    # 输入为指定task_level的城市位置，有指定server_level对应的server位置, 和分配数量，
    # 然后得到城市的位置range和server的位置range， 随机匹配allocate_num的个数量，约束是如果以及分配完的目的地不应该再次分配
    # task_level = 3  # 指定的level
    this_level = f"level {task_level}"
    selected_cities = initial_state_df.loc[initial_state_df[this_level] >= 1][this_level]
    # selected_cities["city 20"] = 3
    # print("selected_cities", selected_cities)
    selected_cities_idx = selected_cities.index
    selected_cities_list = list(selected_cities_idx.str.replace("city ", "").astype(int))
    # print("selected_cities_list", selected_cities_list)

    # server_level = 1
    selected_server_cities = servers_remain_df[servers_remain_df["level"]==server_level]["home"]
    selected_server_list  = list(selected_server_cities)
    selected_cities_idx = selected_server_cities.index
    selected_cities_idx_list = list(selected_cities_idx.str.replace("server ", "").astype(int))

    city_2_idx = {city:idx for city,idx in zip(selected_server_list, selected_cities_idx_list)}
    # print("S len selected_server_list", len(selected_server_list), )
    # 根据值大于2的城市生成对应数量的重复元素
    repeated_cities = list(np.repeat(selected_cities_list, selected_cities.values))

    # print("C len repeated_cities:", len(repeated_cities))

    # 获取业务员所在城市的列表
    # 生成所有城市-业务员位置的排列组合
    # city_salesperson_combinations = list(itertools.product(repeated_cities, selected_server_list))

    all_combinations = []
    min_sum, min_combination = get_combinations(selected_server_list, repeated_cities, [], all_combinations)
    memory_usage = sys.getsizeof(all_combinations)
    len_all_combinations = len(all_combinations)
    print(f"{memory_usage=}(Bytes) {len_all_combinations=} {all_combinations=} ")

    # print(f"Total number of combinations: {len(all_combinations)}")
    # print("all_combinations", all_combinations)

    # 从排列组合中选择指定数量的元素作为任务分配结果
    need_comb_num = allocate_num
    selected_elements = set()

    # 从每个组合中选取两个元素
    if need_comb_num != len(selected_server_list):
        min_sum = float('inf')
        min_combination = None

        for combination in all_combinations:
            pairs = itertools.combinations(combination, need_comb_num)
            current_sum = 0
            for pair in pairs:
                server,city = pair
                current_sum += a_city[server][city]
            if current_sum > min_sum:
                min_sum = current_sum
                min_combination = pair
            # print(f"{pairs=}")
            # 将每个组合中的两个元素添加到选取的元素集合中
            selected_elements.update(pairs)
            
    else:
        selected_elements = all_combinations
    

    print("min Combination:", min_combination)
    print("min Sum:", min_sum)
    final_revenue = revenue - min_sum

    print(f"Total number of selected elements: {len(selected_elements)}")
    memory_usage_selected_elements = sys.getsizeof(selected_elements)
    len_selected_elements = len(selected_elements)
    print(f" {memory_usage_selected_elements=}(Bytes) {len_selected_elements=} {selected_elements=} ")
    # print("selected_elements", selected_elements) # 所有的分配
    # [((城市，server),(城市，server),...),
    #  ((城市，server),(城市，server),...),
    # ...]
    # 分配后应该更新server 位置 为城市
    new_selected_elements = []
    print("（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）")
    # 存到内存
    new_min_combination = []
    for server, city in min_combination:
        new_min_combination.append([city_2_idx[server], server, city, task_level, server_level])
    
    # new_selected_elements = 
    for allocate_group in selected_elements:
        # print("allocate_group", allocate_group)
        new_allocate_group = []
        # for task_city, server_city in zip(allocate_group, selected_server_list):
        for allocate in allocate_group:
            # print(allocate, city_2_idx)
            server_city, task_city  = allocate
            # （业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
            new_allocate = [city_2_idx[server_city], server_city, task_city, task_level, server_level]
            
            # 下面如果保存内存占用太多了！所以应该直接决策的
            new_allocate_group.append(new_allocate)
        new_selected_elements.append(new_allocate_group)
    
    memory_usage_of_new_selected_elements = sys.getsizeof(new_selected_elements)
    
    print(f" {memory_usage_of_new_selected_elements=}(Bytes) {new_selected_elements=} ")
    return new_selected_elements, final_revenue, new_min_combination


def allocate_servers_2_cities(decisions: list[tuple], initial_state_df, servers_remain_df):
     # 分配所有情况
    decision_allocations = []
    decision_allocations_revenue_combination = []
    decisions_num = len(decisions)
    print(f"{decisions_num=}{decisions=}")
    max_revenue = float("-inf")
    max_combination = ()
    for decision in decisions:
        city_servers_for_a_decision_list, final_revenue, min_combination = allocate_servers_2_cities_for_a_decision(decision, initial_state_df, servers_remain_df)
        decision_allocations.append(city_servers_for_a_decision_list)
        decision_allocations_revenue_combination.append((final_revenue, min_combination))
        if final_revenue > max_revenue:
            max_revenue = final_revenue
            max_combination = min_combination
        print(f"{min_combination=}")

    return decision_allocations, decision_allocations_revenue_combination, max_revenue, max_combination


def write_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')
    print(f"{filename=} saved!")

# 根据最优解的combination 求得业务员之后的位置
# （业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
def update_server_cities(servers_df, allocation_for_a_day):
    new_servers_df = servers_df.copy()
    new_servers_df.columns = ["current_city", "level", "day off"]
    for allocation in allocation_for_a_day:
        server_id, server_city, allocate_city, server_level, city_level = allocation[0], allocation[1], allocation[2], allocation[3], allocation[4]
        new_servers_df["current_city"][f"server {server_id}"] = allocate_city
    
    return new_servers_df
# 并求得城市任务数量更新，工作完剩余的任务 + 新生成的任务

def update_task(state_df, allocation_for_a_day):
    global arriving_rate_df
    new_state_df = state_df.copy()
    
    for allocation in allocation_for_a_day:
        server_id, server_city, allocate_city, server_level, city_level = allocation[0], allocation[1], allocation[2], allocation[3], allocation[4]
        if new_state_df[f"level {city_level}"][f"city {allocate_city}"] -1 < 0:
            print(f'ERROR {new_state_df[f"level {city_level}"][f"city {allocate_city}"]}')
        else:
            new_state_df[f"level {city_level}"][f"city {allocate_city}"] -= 1

    arriving_rate_matrix = np.array(arriving_rate_df)  # 将到达率转换为NumPy数组
    current_state_df = state_df# 根据你的实际情况创建当前状态的DataFrame

    # 生成新任务
    tasks = generate_tasks(arriving_rate_matrix)

    # 更新状态矩阵
    new_state_df = update_state(current_state_df, tasks)

    return new_state_df


def generate_tasks(arriving_rate):
    tasks = np.random.poisson(arriving_rate)
    return tasks


def update_state(state_df, tasks):
    updated_state_df = state_df.copy()
    
    tasks_df = pd.DataFrame(tasks, index=state_df.index, columns=state_df.columns)
    for city in state_df.index:
        for level in state_df.columns:
            updated_state_df.loc[city, level] += tasks_df.loc[city, level]
    return updated_state_df


def generate_idx_2_joins(state_df, servers_remain_df):
    
    # 城市对应等级任务矩阵
    city_level_matrix_df = state_df.iloc[:, :]
    # 任务计数，用于划分集合
    task_level_counts = city_level_matrix_df.sum()
    # 业务员等级计数，用于划分集合
    server_level_counts = servers_remain_df['level'].value_counts().sort_index()
    level_num = city_level_matrix_df.shape[0]      # 任务等级数量
    city_num = city_level_matrix_df.shape[0]    # 城市数量
    # 根据levels， servers，tasks区分组别
    servers = list(server_level_counts)
    tasks = list(task_level_counts)
    levels = list(range(1, len(tasks)+1))
    print("levels", levels)
    print("tasks", tasks, "sum(tasks)", sum(tasks))
    print("servers", servers, "sum(servers)", sum(servers))
    joins = generate_joins(task_level_counts, server_level_counts)
    print("joins", joins)
    # 组别内allocate
    join_idx_2_decisions = {}
    for join_idx, join in enumerate(joins):
        level_indices = [levels.index(level) for level in join]  # 获取join中每个level对应的索引
        a_tasks = [tasks[index] for index in level_indices]  # 获取与join中每个level相对应的任务数量
        a_servers = [servers[index] for index in level_indices]  # 获取与join中每个level相对应的服务器数量
        a_levels = [levels[index] for index in level_indices]
        is_done, decisions = allocate(a_tasks, a_servers, a_levels, decision=[])
        join_idx_2_decisions.update({join_idx: decisions})
    # 对每一个decision，对于不同地方和不同的人，进行排列组合分配
    # 获取每个城市情况
    # 每个server情况

    return join_idx_2_decisions




# %%


# %%
global revenue_for_level, a_city
# 收益率
revenue_for_level = [3500, 3000, 2500, 2000, 1500]
np.random.seed(42)
# 生成 26个城市
city, city_num_2_name = generate_city()
city_names = city.columns
a_city, city_num_2_name = change_df_city_name_2_idx(cities=city)

global arriving_rate_df
arriving_rate_df = pd.read_excel('./data/数据.xlsx', sheet_name='arriving rate', index_col=0)
travel_fee_df = pd.read_excel('./data/数据.xlsx', sheet_name='travel fee', index_col=0)
initial_state_df = pd.read_excel('./data/数据.xlsx', sheet_name='initial state', index_col=0)
servers_df = pd.read_excel('./data/数据.xlsx', sheet_name='servers', index_col=0) # 员工

# %%
# # 测试分配
# tasks =  [2, 2, 11]
# servers = [3, 4, 1]
# levels =  [1, 2, 3]
# is_done, decisions = allocate(tasks, servers, levels, decision = [])
# print(decisions)
# for decision in decisions:
#     print(decision)
# A = [(servers_df.loc[i, 'home'], servers_df.loc[i, 'day off']) for i in range(0, len(servers_df))]
# servers_df[servers_df["level"]==1]
# # server_level_counts
# this_level = f"level {3}"
# 筛选得到
# selected_cities = initial_state_df.loc[initial_state_df[this_level] >= 1][this_level]
# selected_cities
# initial_state_df
# 对于每个join，产生其分配方案，生成所有分配方案，分配方案是指
# 当前日子，对于每个城市的状态，业务员的状态，生成一组对业务员的分配，
# 可能为（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
a_state_df = initial_state_df.copy()
a_servers_df = servers_df.copy()
for weekday in range(1, 8):
    # 先排除当天放假的员工
    servers_remain_df = a_servers_df[a_servers_df['day off'] != weekday]
    join_idx_2_decisions = generate_idx_2_joins(a_state_df, servers_remain_df)
    # print("join_idx_2_decisions", join_idx_2_decisions)
    cnt = 0
    # print("（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）")
    join_idx_2_decisions_allocations = []
    # decision_allocations_revenue_combination_for_decisionss = []
    allocation_for_a_day = []
    revenue_sum = 0
    for join_idx, decisions_for_each_joins in join_idx_2_decisions.items():
        cnt += 1
        # 根据集合分组后的，集合内level的数量分配
        print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
        # decisions_for_each_join = [[(1, 1, 4)]]
        all_allocations_for_decisions = []
        # decision_allocations_revenue_combination_for_decisions = []
        for decisions in decisions_for_each_joins:
            # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
            # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
            # 根据数量分配，进行实际的城市分配方案 list下的一个list是
            all_allocations, decision_allocations_revenue_combination, max_revenue, max_combination = allocate_servers_2_cities(decisions, a_state_df, servers_remain_df)
            # all_allocations = [[
            #   ((城市，server),(城市，server),...),
            #   ((城市，server),(城市，server),...),
            # ]]
            all_allocations_for_decisions.append(all_allocations)
            # decision_allocations_revenue_combination_for_decisions.append(decision_allocations_revenue_combination)
            len_all_allocations = len(all_allocations)
            print(f"{len_all_allocations=} {all_allocations=} ")
        
        revenue_sum += max_revenue
        allocation_for_a_day.extend(max_combination)
        
        #decision_allocations_revenue_combination_for_decisionss.append(decision_allocations_revenue_combination_for_decisions)

        
        write_list_to_file(all_allocations_for_decisions, f"all_allocations_for_decisions_{join_idx}_weekday_{weekday}.txt")
        # join_idx_2_decisions_allocations.append(all_allocations_for_decisions)
        #     for a_allocations in all_allocations:
        #         # a_allocations = [((城市，server),(城市，server),...),
        #         #  ((城市，server),(城市，server),...),
        #         # ...]
        #         for allocation in a_allocations:
        #             # allocation = ((城市，server),(城市，server),...)
        #             pass
        # if cnt==3:
        #     break
        # =32984(Bytes)
        # 
    # （业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
    # allocation_for_a_day = []
    # revenue_sum = 0
    # for decision_allocations_revenue_combination_for_decisions in decision_allocations_revenue_combination_for_decisionss:
    #     print(f"{decision_allocations_revenue_combination_for_decisions=}")
    #     for decision_allocations_revenue_combination_for_decision in decision_allocations_revenue_combination_for_decisions:
    #         print(f"{decision_allocations_revenue_combination_for_decision=}")
    #         # 从这里面找到renvene 最大的
    #         max_revenue = float("-inf")
    #         max_combination = ()
    #         for revenue, combination in decision_allocations_revenue_combination_for_decision:
    #             if revenue > max_revenue:
    #                 max_revenue = revenue
    #                 max_combination = combination
    #             print(f"{combination=}")
    #         revenue_sum += max_revenue
    #         allocation_for_a_day.extend(max_combination)

    # print(revenue_sum, allocation_for_a_day)
    # allocate_servers_2_cities()

    new_state_df = a_state_df.copy()
    for allocation in allocation_for_a_day:
        server_id, server_city, allocate_city, server_level, city_level = allocation[0], allocation[1], allocation[2], allocation[3], allocation[4]
        if new_state_df[f"level {city_level}"][f"city {allocate_city}"] -1 < 0:
            print(f'ERROR {new_state_df[f"level {city_level}"][f"city {allocate_city}"]}')
        else:
            new_state_df[f"level {city_level}"][f"city {allocate_city}"] -= 1

    # arriving_rate_array = arriving_rate_df.values
    arriving_rate_matrix = arriving_rate_df.values # 将到达率转换为NumPy数组
    current_state_df = a_state_df# 根据你的实际情况创建当前状态的DataFrame
    # print(type(arriving_rate_matrix[0]), arriving_rate_matrix[0][0])
    # 生成新任务
    tasks = generate_tasks(arriving_rate_matrix)
    # 更新状态矩阵
    new_state_df = update_state(current_state_df, tasks)
    new_servers_df = update_server_cities(servers_df, allocation_for_a_day)

    a_state_df = new_state_df
    a_servers_df = new_servers_df

# %%
# # # 测试分配
# # tasks =  [2, 2, 11]
# # servers = [3, 4, 1]
# # levels =  [1, 2, 3]
# # is_done, decisions = allocate(tasks, servers, levels, decision = [])
# # print(decisions)
# # for decision in decisions:
# #     print(decision)
# # A = [(servers_df.loc[i, 'home'], servers_df.loc[i, 'day off']) for i in range(0, len(servers_df))]
# # servers_df[servers_df["level"]==1]
# # # server_level_counts
# # this_level = f"level {3}"
# # 筛选得到
# # selected_cities = initial_state_df.loc[initial_state_df[this_level] >= 1][this_level]
# # selected_cities
# # initial_state_df
# # 对于每个join，产生其分配方案，生成所有分配方案，分配方案是指
# # 当前日子，对于每个城市的状态，业务员的状态，生成一组对业务员的分配，
# # 可能为（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
# a_state_df = initial_state_df.copy()
# a_servers_df = servers_df.copy()
# for weekday in range(1, 8):
#     # 先排除当天放假的员工
    
#     join_idx_2_decisions_allocations = []
#     decision_allocations_revenue_combination_for_decisionss = []

#     servers_remain_df = a_servers_df[a_servers_df['day off'] != weekday]
#     join_idx_2_decisions = generate_idx_2_joins(a_state_df, servers_remain_df) # 生成不同level集合组内的分配方式 用idx作为索引可取出分配
#     # print("（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）")

#     for join_idx, decisions_for_each_joins in join_idx_2_decisions.items():
#         # 根据集合分组后的，集合内level的数量分配 decisions_for_each_join = [[(1, 1, 4)]]
#         print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)

#         all_allocations_for_decisions = []
#         decision_allocations_revenue_combination_for_decisions = []
#         for decisions in decisions_for_each_joins:
#             # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
#             all_allocations, decision_allocations_revenue_combination = allocate_servers_2_cities(decisions, a_state_df, servers_remain_df) # 根据数量分配，进行实际的城市分配方案 
#             # all_allocations = [[
#             # ((城市，server),(城市，server),...),
#             # ((城市，server),(城市，server),...),
#             # ]]
#             all_allocations_for_decisions.append(all_allocations)
#             decision_allocations_revenue_combination_for_decisions.append(decision_allocations_revenue_combination)
#             len_all_allocations = len(all_allocations)
#             print(f"{len_all_allocations=} {all_allocations=} ")
#         decision_allocations_revenue_combination_for_decisionss.append(decision_allocations_revenue_combination_for_decisions)
#             # break
        
#         write_list_to_file(all_allocations_for_decisions, f"all_allocations_for_decisions_{join_idx}.txt")
#         # join_idx_2_decisions_allocations.append(all_allocations_for_decisions)
#         #     for a_allocations in all_allocations:
#         #         # a_allocations = [((城市，server),(城市，server),...),
#         #         #  ((城市，server),(城市，server),...),
#         #         # ...]
#         #         for allocation in a_allocations:
#         #             # allocation = ((城市，server),(城市，server),...)
#         #             pass
#         if weekday==4:
#             break
#         # =32984(Bytes)
#         # 
#     # （业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）
#     allocation_for_a_day = []
#     revenue_sum = 0
#     for decision_allocations_revenue_combination_for_decisions in decision_allocations_revenue_combination_for_decisionss:
#         print(f"{decision_allocations_revenue_combination_for_decisions=}")
#         for decision_allocations_revenue_combination_for_decision in decision_allocations_revenue_combination_for_decisions:
#             print(f"{decision_allocations_revenue_combination_for_decision=}")
#             # 从这里面找到renvene 最大的
#             max_revenue = float("-inf")
#             max_combination = ()
#             for revenue, combination in decision_allocations_revenue_combination_for_decision:
#                 if revenue > max_revenue:
#                     max_revenue = revenue
#                     max_combination = combination
#                 print(f"{combination=}")
#             revenue_sum += max_revenue
#             allocation_for_a_day.extend(max_combination)

#     print(revenue_sum, allocation_for_a_day)
#     # allocate_servers_2_cities()


#     # servers_df = initial_state_df
#     a_state_df = initial_state_df
#     new_state_df = a_state_df.copy()
#     for allocation in allocation_for_a_day:
#         server_id, server_city, allocate_city, server_level, city_level = allocation[0], allocation[1], allocation[2], allocation[3], allocation[4]
#         if new_state_df[f"level {city_level}"][f"city {allocate_city}"] -1 < 0:
#             print(f'ERROR {new_state_df[f"level {city_level}"][f"city {allocate_city}"]}')
#         else:
#             new_state_df[f"level {city_level}"][f"city {allocate_city}"] -= 1

#     # arriving_rate_array = arriving_rate_df.values
#     arriving_rate_matrix = arriving_rate_df.values # 将到达率转换为NumPy数组
#     current_state_df = a_state_df# 根据你的实际情况创建当前状态的DataFrame
#     print(type(arriving_rate_matrix[0]), arriving_rate_matrix[0][0])
#     # 生成新任务
#     tasks = generate_tasks(arriving_rate_matrix)
#     # 更新状态矩阵
#     new_state_df = update_state(current_state_df, tasks)


