import pandas as pd
import numpy as np
from allocate_cls import allocate
def allocate_servers_2_cities_for_a_decision(allocate_tuple, initial_state_df, servers_remain_df):
    global revenue_for_lv, a_city_distance_df
    
    task_lv, server_lv, allocate_num = allocate_tuple
    # if allocate_num > 1:
    revenue = revenue_for_lv[task_lv-1] * allocate_num
    cost = 0
    
    # servers_remain_df
    # 对于一个decisions 内的单个allocation，进行分配
    # 输入为指定task_lv的城市位置，有指定server_lv对应的server位置, 和分配数量
    # 然后得到城市的位置range和server的位置range， 随机匹配allocate_num的个数量，约束是如果以及分配完的目的地不应该再次分配
    # task_lv = 3  # 指定的level
    this_lv = f"lv {task_lv}"
    selected_task_city_df = initial_state_df.loc[initial_state_df[this_lv] >= 1][this_lv] # 选择出当前 lv 满足task_lv要求的城市
    
    selected_task_city_idx = selected_task_city_df.index # task 城市idx
    selected_task_city_id_list = list(selected_task_city_idx.str.replace("city ", "").astype(int)) # task 城市id
    # 根据值大于2的城市生成对应数量的重复元素，用于计算分配
    repeated_city = list(np.repeat(selected_task_city_id_list, selected_task_city_df.values))

    # server_lv = 1
    selected_server_city_df = servers_remain_df[servers_remain_df["lv"]==server_lv]["current_city"]# 选择 lv 满足 server_lv 的server 
    selected_server_city_id_list  = list(selected_server_city_df)        # server 城市 id
    selected_server_idx = selected_server_city_df.index   # server idx
    selected_server_id_list = list(selected_server_idx.str.replace("server ", "").astype(int))# server id

    server_city_id_2_server_id = {city:idx for city,idx in zip(selected_server_city_id_list, selected_server_id_list)} # 城市id转业务员id，用于生成分配策略
    

    # 获取业务员所在城市的列表
    # 生成所有城市-业务员位置的排列组合

    # 获取 满足task_lv要求的城市 和 满足 server_lv 的server 条件下的所有分配，
    min_sum, min_combination = get_combinations(selected_server_city_id_list, repeated_city, [])
        
    # else:
    #     min_combination = [(task_lv, server_lv)]
    #     min_sum  = 
    print("min Combination:", min_combination) # cost最小的组合
    print("min Sum:", min_sum)
    final_revenue = revenue - min_sum 

    # 分配后应该更新server 位置 为城市
    print("（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）")

    # 存到内存
    new_min_combination = []
    if min_combination:
        print(min_combination)
    else:
        print(f"{min_combination=}")
    for server, city in min_combination:
        new_min_combination.append([server_city_id_2_server_id[server], server, city, task_lv, server_lv])
    return final_revenue, new_min_combinations


def allocate_servers_2_cities(decisions: list[tuple], initial_state_df, servers_remain_df)->list[dict]:
     # 根据decisions集合分组情况/城市状态df/用户状态df，分配所有情况
     
    revenue_and_combination_for_decisions = []
    decisions_num = len(decisions)
    print(f"{decisions_num=} {decisions=}")
    max_revenue = float("-inf")
    max_combination = ()
    # 对于组内的decision 返回的应该是多种分配
    for decision in decisions:
        final_revenue, min_combination = allocate_servers_2_cities_for_a_decision(decision, initial_state_df, servers_remain_df)
        revenue_and_combination_for_decisions.append({'revenue':final_revenue, 'combination':min_combination})
        print(f"{final_revenue=} {min_combination=}")


    return revenue_and_combination_for_decisions


def cul_a_cycle(T, a_servers_df, a_state_df):
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T+1):
    
        # ------根据 lv 来做分组，进行组内分配
        print(f"-----------------------------------------------weekday:{weekday}------------------------------------------")
        # 先排除当天放假的员工
        servers_remain_df = a_servers_df[a_servers_df['day off'] != weekday]
        print(f"{servers_remain_df=}")
        # 根据 lv 来做分组, join 分组后的结果应该
        join_idx_2_decisions = generate_idx_2_joins(a_state_df, servers_remain_df)
        print(f"{join_idx_2_decisions=}")
        cnt = 0
        # print("（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级）")
        
        join_idx_2_decisions_allocations = []
        allocation_for_a_day = []
        revenue_sum = 0
        # 对于每一个集合，进行集合内分配
        for join_idx, decisions_for_each_joins in join_idx_2_decisions.items():
            cnt += 1
            # 根据集合分组后的，集合内level的数量分配
            print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
            # decisions_for_each_join = [[(1, 1, 4)]]
            all_best_allocations_for_decisions = {}

            for idx, decisions in enumerate(decisions_for_each_joins):
                # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
                # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
                # 根据数量分配，进行实际的城市分配方案 list下的一个list是

                revenue_and_combination_for_decisions = allocate_servers_2_cities(decisions, a_state_df, servers_remain_df)
                # revenue_and_combination_for_decisions = {'revenue':final_revenue, 'combination':min_combination}
                all_best_allocations_for_decisions.update({idx:revenue_and_combination_for_decisions})

            # 开始合并决策，并reduce 并获取V历史记录的对应的收益。
            num_combinations, revenue_and_combinations = get_all_allocations_for_decisions(all_best_allocations_for_decisions, a_servers_df, a_state_df)

        print("test")

        new_state_df = a_state_df.copy()
        for allocation in allocation_for_a_day:
            server_id, server_city, allocate_city, server_lv, city_lv = allocation[0], allocation[1], allocation[2], allocation[3], allocation[4]
            if new_state_df[f"lv {city_lv}"][f"city {allocate_city}"] -1 < 0:
                print(f'ERROR {new_state_df[f"lv {city_lv}"][f"city {allocate_city}"]}')
            else:
                new_state_df[f"lv {city_lv}"][f"city {allocate_city}"] -= 1

        arriving_rate_matrix = arriving_rate_df.values # 将到达率转换为NumPy数组
        current_state_df = a_state_df# 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)
        # 更新状态矩阵
        new_state_df = update_state(current_state_df, new_tasks)
        new_servers_df = update_server_cities(a_servers_df, allocation_for_a_day)

        a_state_df = new_state_df
        a_servers_df = new_servers_df
# 先判断是否可以分配，划分分配的区间，区间内任意分配，
def generate_joins(task_lv_count:pd.core.series.Series, server_lv_count:pd.core.series.Series)->list[list]:
    # 输入
    # task_lv_count pandas.core.series.Series(level_num,) 任务不同等级
    # server_lv_count pandas.core.series.Series(level_num,)
    # server number tag task number
    # lv 1 6 > 5
    # lv 2 2 > 1
    # lv 3 3 ≤ 6
    # lv 4 2 > 1
    # lv 5 4 ≤ 17
    # lv 6 5 > 4
    # lv 7 7 > 3
    # 需要根据不同等级任务和员工的等级做个计数判断，当连续的每个等级各自同时加起来大于现有的任务等级，表示这一组level之前可以任意分配，可以分配出不同的决策，
    # 比如上面的表里level1和level 2 server number > task number
    # lv 1可以分配给level2，也可以level2 都分配给level2，这是不同的决策空间，
    # 但是level 1 task也必须被做完。当level3 也考虑进去的时候，123相加就server数量就比task少了
    # 这时候level4 就不能和123 一起考虑，但是123 之间可以一起决策，level3 全部分配到task
    # 同时level 2 个也得分配到level3 或者level2
    # 输出 return joins = [[1,2,3],[4],[5]]
    # 输入校验：
    if task_lv_count.shape[0] != server_lv_count.shape[0]:
        raise Exception("输入pandas.core.series.Series形状不一致")

    joins = []  # 存储决策空间的列表
    current_join = []  # 当前的决策空间划分
    task_lv_num = task_lv_count.shape[0]      # 任务等级数量 
    task_join_count = 0
    server_join_count = 0

    for idx, (task_count, server_count) in enumerate(zip(task_lv_count, server_lv_count)):
        task_join_count += task_count
        server_join_count += server_count
        lv = idx+1
        if server_join_count <= task_join_count:
            # 当任务数量大于员工数量时，将当前决策空间划分添加到列表中
            if current_join:
                current_join.append(lv)
                joins.append(current_join)
            else:
                joins.append([lv])
                
            current_join = []  # 重置当前决策空间划分
            task_join_count = 0
            server_join_count = 0
        else:
            # 当任务数量小于等于员工数量时，将当前等级添加到当前决策空间划分中
            current_join.append(lv)
    if current_join:
        joins.append(current_join)  # 将最后一个决策空间划分添加到列表中
    return joins


def generate_idx_2_joins(state_df, servers_remain_df):
    
    # 城市对应等级任务矩阵
    city_lv_matrix_df = state_df.iloc[:, :]
    # 任务计数，用于划分集合
    task_lv_count = city_lv_matrix_df.sum()
    # 业务员等级计数，用于划分集合
    server_lv_count = servers_remain_df['lv'].value_counts().sort_index()
    level_num = city_lv_matrix_df.shape[0]      # 任务等级数量
    city_num = city_lv_matrix_df.shape[0]    # 城市数量
    # 根据levels， servers，tasks区分组别
    servers = list(server_lv_count)
    tasks = task_lv_count.to_list()
    levels = list(range(1, len(tasks)+1))
    print(f"{levels=}", )
    print(f"{tasks=} {sum(tasks)=}")
    print(f"{servers=} {sum(servers)=}")
    joins = generate_joins(task_lv_count, server_lv_count)
    print(f"根据不同level的task数量和server数量, 组内数量分配结果如下 {joins=}")
    # 组别内allocate
    join_idx_2_decisions = {}
    for join_idx, join in enumerate(joins):
        level_indices = [levels.index(lv) for lv in join]  # 获取join中每个level对应的索引
        a_tasks = [tasks[index] for index in level_indices]  # 获取与join中每个level相对应的任务数量
        a_servers = [servers[index] for index in level_indices]  # 获取与join中每个level相对应的服务器数量
        a_lvs = [levels[index] for index in level_indices]
        decisions = allocate(a_tasks, a_servers, a_lvs)
        join_idx_2_decisions.update({join_idx: decisions})
    # 对每一个decision，对于不同地方和不同的人，进行排列组合分配
    # 获取每个城市情况
    # 每个server情况

    return join_idx_2_decisions



# 2024年1月21日 21点42分
def change_df_city_name_2_idx(cities: pd.DataFrame)->(pd.DataFrame, dict):
    """改名字为idx，同时返回新df和idx2name"""
    city_names = cities.columns
    city_num = len(city_names)
    city_name_nums = list(range(1, city_num+1))
    city_num_2_name = {key:value for key,value in zip(city_name_nums, city_names)}
    new_cities = cities.copy()
    new_cities.columns = new_cities.index = city_name_nums
    return new_cities, city_num_2_name

# 2024年1月21日 20点42分
def get_proveng_city_dist_mat_df():
    proveng_city_dist_mat_df = pd.read_excel(r'D:\Users\sjc\algorithm\paper\data\中国各城市空间权重矩阵(1).xlsx', sheet_name='地理距离矩阵')
    return proveng_city_dist_mat_df


def get_city_2_proveng_dict():
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_proveng_dict = proveng_city_dist_mat_df[['cityeng', 'proveng']].drop_duplicates().set_index('cityeng')['proveng'].to_dict()
    return city_to_proveng_dict

def get_city_series_to_city_dict():
    
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_series_to_city_dict = proveng_city_dist_mat_df[['cityeng', 'cityseries']].drop_duplicates().set_index('cityseries')['cityeng'].to_dict()
    return city_series_to_city_dict


def get_city_to_city_series_dict():
    
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_city_series_dict = proveng_city_dist_mat_df[['cityeng', 'cityseries']].drop_duplicates().set_index('cityeng')['cityseries'].to_dict()
    return city_to_city_series_dict

def get_dist_mat_from_xls_df(proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx):
    
    distance_mat = proveng_city_dist_mat_df.iloc[:, distance_mat_start_idx:]
    distance_mat.index = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values
    distance_mat.columns = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values

    return distance_mat

def get_select_city_df(city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat):
    # 挑选出安徽、江苏、浙江和上海的省份及对应的矩阵数据
    select_proveng = ['Anhui', 'Jiangsu', 'Zhejiang', 'Shanghai'] # TODO 可能要换省份
    provengs = proveng_city_dist_mat_df.iloc[:, proveng_idx].values # 表格第一列为城市名称
    select_proveng_idxs = [i for i, prov in enumerate(provengs) if prov in select_proveng] # xls里选取指定的省份的idx，是字母
    select_cityeng_idxs = proveng_city_dist_mat_df.values[select_proveng_idxs, cityseries_idx]# xls指定省份对应的城市id
    select_idx = pd.Index(select_cityeng_idxs) # 创建对应索引
    # 使用 loc 方法选择相应的行和列
    select_proveng_city_df = distance_mat.loc[select_idx, select_idx]

    
    # 从城市列表中随机选择city_num-1个城市（不包括上海）
    rnd_cities = np.random.choice(select_cityeng_idxs[1:], city_num-1, replace=False) # 第一列为上海，所以从1开始
    # 将上海添加到随机选择的城市列表中 # City158 = shanghai
    select_cities = ['City158'] + rnd_cities.tolist()
    select_city_idx = pd.Index(select_cities)
    # 使用 loc 方法选择相应的行和列
    select_city_df = select_proveng_city_df.loc[select_city_idx, select_city_idx]

    return select_city_df


def generate_city(city_num:int=26)->(pd.DataFrame, dict):
    '''本函数功能，生成指定数量城市的城市距离矩阵，必然包含上海。返回的dataframe包含城市坐标和省份坐标/和一个城市转省份的dict'''

    # 地理距离矩阵xls
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    # 从第四列开始，是距离矩阵内容
    proveng_idx = 0
    cityeng = 1
    cityseries_idx = 2
    distance_mat_start_idx = 3

    # 
    city_series_to_city_dict = get_city_series_to_city_dict()
    city_to_proveng = get_city_2_proveng_dict()

    # 地理距离矩阵xls 第三列是城市id, city_*, 设定纵坐标
    distance_mat = get_dist_mat_from_xls_df(proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx)
    select_city_df = get_select_city_df(city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat)


    city_series_columns = select_city_df.columns
    city_columns = [(city_to_proveng[city_series_to_city_dict[item]], city_series_to_city_dict[item]) for item in city_series_columns ] # 最为新矩阵的idx和col
    

    column_names = ['proveng', 'cityeng'] # 给idx的idx
    city_df = pd.DataFrame(select_city_df.values, 
                        index=pd.MultiIndex.from_tuples(city_columns, names=column_names), 
                        columns=pd.MultiIndex.from_tuples(city_columns, names=column_names))
    return city_df


if __name__ == "__main__":
    city = generate_city(city_num=10)
    city_to_proveng = get_city_2_proveng_dict()