import itertools
from collections import Counter

import numpy as np
import pandas as pd
from allocate_cls import allocate
import random


def get_combinations(
    server_city_id_list,
    task_city_id_list,
    current_combination,
    min_cost_sum=float("inf"),
    allocate_else=0,
    min_cost_city_id_of_server_and_task_combination_list=[],
    need_comb_num=None,
    a_city_distance_df=None,
):
    if (
        len(server_city_id_list) == 0
        or len(task_city_id_list) == 0
        # or allocate_else == 0
    ):
        assert current_combination != None
        # TODO 取消存储防止爆内存
        current_sum = sum(
            [a_city_distance_df[server][city] for server, city in current_combination]
        )
        if current_sum < min_cost_sum:
            min_cost_sum = current_sum
            min_cost_city_id_of_server_and_task_combination_list = [current_combination]
        # elif current_sum == min_cost_sum:
        #     min_cost_city_id_of_server_and_task_combination_list.append(
        #         current_combination
        #     )
        if need_comb_num is not None:
            # Filter min_cost_city_id_of_server_and_task_combination to contain only the first need_comb_num combinations
            pairs = itertools.combinations(
                min_cost_city_id_of_server_and_task_combination, need_comb_num
            )
            new_min_cost_sum = float("inf")
            new_min_cost_city_id_of_server_and_task_combination = None
            for pair in pairs:
                current_sum = sum(
                    [a_city_distance_df[server][city] for server, city in pair]
                )
                if current_sum < new_min_cost_sum:
                    new_min_cost_sum = current_sum
                    new_min_cost_city_id_of_server_and_task_combination = pair
            min_cost_sum = new_min_cost_sum
            min_cost_city_id_of_server_and_task_combination = (
                new_min_cost_city_id_of_server_and_task_combination
            )
        return min_cost_sum, min_cost_city_id_of_server_and_task_combination_list

    for i, city in enumerate(task_city_id_list):
        remaining_cities = task_city_id_list[:i] + task_city_id_list[i + 1 :]
        new_current_combination = current_combination + [(server_city_id_list[0], city)]
        (
            min_cost_sum,
            min_cost_city_id_of_server_and_task_combination_list,
        ) = get_combinations(
            server_city_id_list[1:],
            remaining_cities,
            new_current_combination,
            min_cost_sum,
            allocate_else - 1,
            min_cost_city_id_of_server_and_task_combination_list,
            need_comb_num,
            a_city_distance_df,
        )

    return min_cost_sum, min_cost_city_id_of_server_and_task_combination_list


def get_revenue_by_task_and_num(task_lv, num):
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    return revenue_for_lv[task_lv - 1] * num


def get_selected_task_lv_city_id_list(task_lv, a_task_df):
    # remain_servers_df
    # 对于一个decisions 内的单个allocation，进行分配
    # 输入为指定task_lv的城市位置，有指定server_lv对应的server位置, 和分配数量
    # 然后得到城市的位置range和server的位置range， 随机匹配allocate_num的个数量，约束是如果以及分配完的目的地不应该再次分配
    # task_lv = 3  # 指定的level
    this_lv = f"level {task_lv}"
    selected_task_city_df = a_task_df.loc[a_task_df[this_lv] >= 1][
        this_lv
    ]  # 选择出当前 lv 满足task_lv要求的城市

    selected_task_city_idx = selected_task_city_df.index  # task 城市idx
    selected_task_city_id_list = list(
        selected_task_city_idx.str.replace("city ", "").astype(int)
    )  # task 城市id
    # 根据值大于2的城市生成对应数量的重复元素，用于计算分配
    selected_task_city_id_list_repeated = list(
        np.repeat(selected_task_city_id_list, selected_task_city_df.values)
    )
    return selected_task_city_id_list_repeated


def get_selected_selected_server_city_n_id_list(server_lv, remain_servers_df):
    # server_lv = 1
    selected_server_city_df = remain_servers_df[remain_servers_df["lv"] == server_lv][
        "current_city"
    ]  # 选择 lv 满足 server_lv 的server
    selected_server_city_id_list = list(selected_server_city_df)  # server 城市 id
    selected_server_idx = selected_server_city_df.index  # server idx
    selected_server_id_list = list(
        selected_server_idx.str.replace("server ", "").astype(int)
    )  # server id

    return selected_server_id_list, selected_server_city_id_list


# 任务等级，业务员等级，数量
# task_lv, server_lv, allocate_num
# 分配指定等级对应的城市和业务员位置
def allocate_servers_2_cities_for_a_decision(
    allocate_tuple, a_task_df, remain_servers_df, a_city_distance_df
):
    #
    task_lv, server_lv, allocate_num = allocate_tuple
    # if allocate_num > 1:
    # 完成tasklv的任务可以获得的奖励
    revenue = get_revenue_by_task_and_num(task_lv, allocate_num)
    min_cost_sum = 0

    # 获取指定level为task_lv的task的城市id
    selected_task_city_id_list_repeated = get_selected_task_lv_city_id_list(
        task_lv=task_lv, a_task_df=a_task_df
    )
    # 获取指定level为server_lv的server的城市id
    (
        selected_server_id_list,
        selected_server_city_id_list,
    ) = get_selected_selected_server_city_n_id_list(
        server_lv=server_lv, remain_servers_df=remain_servers_df
    )
    #
    server_city_id_2_server_id = {
        city: idx
        for city, idx in zip(selected_server_city_id_list, selected_server_id_list)
    }  # 城市id转业务员id，用于生成分配策略

    # 获取业务员所在城市的列表
    # 生成所有城市-业务员位置的排列组合
    # 获取 满足task_lv要求的城市 和 满足 server_lv 的server 条件下的所有分配，
    (
        min_cost_sum,
        min_cost_city_id_of_server_and_task_combination_list,
    ) = get_combinations(
        selected_server_city_id_list,
        selected_task_city_id_list_repeated,
        current_combination=[],
        a_city_distance_df=a_city_distance_df,
        allocate_else=allocate_num,
    )
    # else:
    #     min_cost_city_id_of_server_and_task_combination = [(task_lv, server_lv)]
    #     min_cost_sum  =
    # print("min Combination:", min_cost_city_id_of_server_and_task_combination) # cost最小的组合
    # print("min Sum:", min_cost_sum)
    final_revenue = revenue - min_cost_sum

    # 分配后应该更新server 位置 为城市
    # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']

    # 存到内存
    new_min_cost_city_id_of_server_and_task_combination_list = []
    # if min_cost_city_id_of_server_and_task_combination:
    #     print(min_cost_city_id_of_server_and_task_combination)
    # else:
    #     print(f"{min_cost_city_id_of_server_and_task_combination=}")
    for (
        min_cost_city_id_of_server_and_task_combination
    ) in min_cost_city_id_of_server_and_task_combination_list:
        new_min_cost_city_id_of_server_and_task_combination = []
        for server_city_id, city_id in min_cost_city_id_of_server_and_task_combination:
            new_min_cost_city_id_of_server_and_task_combination.append(
                [
                    server_city_id_2_server_id[server_city_id],
                    server_city_id,
                    city_id,
                    task_lv,
                    server_lv,
                ]
            )
            # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        new_min_cost_city_id_of_server_and_task_combination_list.append(
            new_min_cost_city_id_of_server_and_task_combination
        )

    return final_revenue, new_min_cost_city_id_of_server_and_task_combination_list


def allocate_servers_2_cities(
    a_decision: list[tuple],
    initial_task_df,
    remain_servers_df,
    a_city_distance_df,
) -> dict:
    # 根据decisions集合分组情况/城市状态df/用户状态df，分配所有情况

    revenue_and_combination_for_decision = {}

    # 对于组内的decision 返回的应该是多种分配
    revenue_sum = 0
    all_combination_for_a_decision = []
    a_task_df = initial_task_df.copy()

    allocate_servers_2_cities_for_a_decision_list = [
        allocate_servers_2_cities_for_a_decision(
            a_allocate,
            # (4, 2, 1),
            a_task_df,
            remain_servers_df,
            a_city_distance_df,
        )
        for a_allocate in a_decision
    ]
    # todo 这里现在是选择了0，但是需要笛卡尔积获得所有情况
    # 然后获取所有分配情况后，生成分配获取收益
    revenue_sum = sum(
        [item[0] for item in allocate_servers_2_cities_for_a_decision_list]
    )  # 对首个加和
    all_combination_for_a_decision = []
    for (
        a_revenue_for_a_allocate,
        combination_for_a_allocate_list,
    ) in allocate_servers_2_cities_for_a_decision_list:
        combination_for_a_allocate = combination_for_a_allocate_list[0]
        if len(combination_for_a_allocate_list) > 1:
            print(combination_for_a_allocate_list)
        all_combination_for_a_decision += combination_for_a_allocate

    # for a_allocate in a_decision:
    #     (
    #         a_revenue_for_a_allocate,
    #         combination_for_a_allocate,
    #     ) = allocate_servers_2_cities_for_a_decision(
    #         a_allocate, a_task_df, remain_servers_df, a_city_distance_df
    #     )
    #     print(a_allocate, a_revenue_for_a_allocate, combination_for_a_allocate)
    #     revenue_sum += a_revenue_for_a_allocate
    #     all_combination_for_a_decision.extend(combination_for_a_allocate)

    revenue_and_combination_for_decision = {
        "revenue": revenue_sum,
        "combination": all_combination_for_a_decision,
    }

    return revenue_and_combination_for_decision


def get_all_allocations_for_decisions(allocations_for_decisions: dict) -> list[dict]:
    num_combinations = 1
    for key, value in allocations_for_decisions.items():
        num_combinations *= len(value)

    # Print the number of possible combinations
    print(f"{num_combinations=}")

    # Get all possible combinations of values from the dictionary
    join_combination = list(itertools.product(*allocations_for_decisions.values()))
    #
    final_revenue_n_combination_list = []
    for a_join_comb in join_combination:
        final_revenue = 0
        final_combination = []

        for idx, join_idx in enumerate(a_join_comb):
            final_revenue += allocations_for_decisions[idx][join_idx]["revenue"]
            final_combination.extend(
                allocations_for_decisions[idx][join_idx]["combination"]
            )
        final_revenue_n_combination_list.append(
            {"final_revenue": final_revenue, "final_combination": final_combination}
        )
    return final_revenue_n_combination_list


def generate_tasks(arriving_rate):
    tasks = np.random.poisson(arriving_rate)
    return tasks


def update_state(state_df, tasks):
    updated_task_df = state_df.copy()

    tasks_df = pd.DataFrame(tasks, index=state_df.index, columns=state_df.columns)
    for city in state_df.index:
        for lv in state_df.columns:
            updated_task_df.loc[city, lv] += tasks_df.loc[city, lv]
    return updated_task_df


def reduce_task_df(a_task_df, proveng_dict, city_num_2_name):
    # 默认是有两列index，一列省份一列市的，缩减为省份的
    new_index = [
        proveng_dict[city_num_2_name[int(city[5:])][0]] for city in a_task_df.index
    ]

    # 将新的索引列表赋值给a_task_df的索引
    reduced_task_df = a_task_df.copy()
    reduced_task_df.index = new_index

    # print("New a_task_df:")
    # print(f"{reduced_task_df=}")
    final_reduced_task_df = reduced_task_df.groupby(reduced_task_df.index).sum()

    return final_reduced_task_df


def reduce_server_df(a_servers_df, proveng_dict, city_num_2_name):
    # 默认是有两列index，一列省份一列市的，缩减为省份的
    reduced_servers_df = a_servers_df.copy()
    reduced_servers_df["current_prov"] = (
        reduced_servers_df["current_city"]
        .map(city_num_2_name)
        .map(lambda x: proveng_dict[x[0]])
    )
    city_prov_tuples = [
        (int(idx[7:]), row["current_prov"])
        for idx, row in reduced_servers_df.iterrows()
    ]

    return city_prov_tuples


# 根据最优解的combination 求得业务员之后的位置
# columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
def update_server_cities(servers_df, allocation_for_a_day):
    new_servers_df = servers_df.copy()
    new_servers_df.columns = ["current_city", "lv", "day off"]
    for allocation in allocation_for_a_day:
        server_id, server_org_city, server_to_city, city_lv, server_lv = (
            allocation[0],
            allocation[1],
            allocation[2],
            allocation[3],
            allocation[4],
        )
        new_servers_df["current_city"][f"server {server_id}"] = server_to_city

    return new_servers_df


def get_allocation_for_a_day(final_revenue_n_combination_list,
                             remain_servers_df, a_task_df,
                             arriving_rate_df, weekday, proveng_dict, city_num_2_name,reduce_V
                             ) -> (float, list[tuple]):
    allocation_for_a_day = []
    # 做决策了这里就是要，挑一个最大的，同时要吧state的值压缩了之后做保存，当前state情况的最优决策
    max_revenue = 0
    max_combination = []
    for revenue_n_combination in final_revenue_n_combination_list:
        revenue = revenue_n_combination["final_revenue"]
        combination = revenue_n_combination["final_combination"]

        a_servers_df = remain_servers_df
        final_allocation_for_a_day = combination
        reduce_V, new_task_df, new_servers_df,allocate_task_df = save_reduct_v(a_task_df, a_servers_df, final_allocation_for_a_day, arriving_rate_df, weekday, proveng_dict, city_num_2_name, revenue, reduce_V)
        reduceV_revenue = get_a_state_revenue(
            a_servers_df,
            new_servers_df,
            a_task_df,
            allocate_task_df,
            reduce_V,
            weekday,
            proveng_dict,
            city_num_2_name,
        )
    
        if max_revenue < revenue+reduceV_revenue:
            max_revenue = revenue+reduceV_revenue
            max_combination = combination
    
    allocation_for_a_day = max_combination
    return max_revenue, allocation_for_a_day


def allcocate_comb_2_allocate_task_df(
    final_allocation_for_a_day: list[tuple], org_task_df: pd.DataFrame
) -> pd.DataFrame:
    new_task_df = org_task_df.copy()
    new_task_df.fillna(0, inplace=True)
    # columns = ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
    for allocation in final_allocation_for_a_day:
        city_num = allocation[2]
        city_level = allocation[3]
        new_task_df.loc[f"city {city_num}", f"level {city_level}"] = 1

    return new_task_df


def allocate_servers_2_citys_MDP(remain_servers_df, a_task_df, a_city_distance_df, arriving_rate_df, weekday, proveng_dict, city_num_2_name,reduce_V):
    # 根据 lv 来做分组, join 分组后的结果应该
    join_idx_2_task_lv_server_lv_num = generate_idx_2_joins(
        a_task_df, remain_servers_df
    )
    # print(f"{join_idx_2_task_lv_server_lv_num=}")
    cnt = 0
    # print("（业务员编号id，业务员城市，分配去的城市编号，城市等级，业务员等级）")

    join_idx_2_revenue_and_combination_list_dict = {}
    revenue_sum = 0
    # 对于每一个集合，进行集合内分配
    for (
        join_idx,
        decisions_for_each_joins,
    ) in join_idx_2_task_lv_server_lv_num.items():
        cnt += 1
        # 根据集合分组后的，集合内level的数量分配
        # print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
        # decisions_for_each_join = [[(1, 1, 4)]]
        revenue_and_combination_list_for_a_join = {}

        for idx, a_decision in enumerate(decisions_for_each_joins):
            # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
            # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
            # 根据数量分配，进行实际的城市分配方案 list下的一个list是
            # print(f"{idx=} {a_decision=}")
            revenue_and_combination_for_decision = allocate_servers_2_cities(
                a_decision, a_task_df, remain_servers_df, a_city_distance_df
            )
            # revenue_and_combination_for_decision = {'revenue':final_revenue, 'combination':min_cost_city_id_of_server_and_task_combination}
            revenue_and_combination_list_for_a_join.update(
                {idx: revenue_and_combination_for_decision}
            )

        join_idx_2_revenue_and_combination_list_dict.update(
            {join_idx: revenue_and_combination_list_for_a_join}
        )

    # 有了不同的，开始合并决策，并reduce 并获取V历史记录的对应的收益。
    final_revenue_n_combination_list = get_all_allocations_for_decisions(
        join_idx_2_revenue_and_combination_list_dict
    )

    # 获取最好的分配
    final_revenue, final_allocation_for_a_day = get_allocation_for_a_day(
        final_revenue_n_combination_list,
        remain_servers_df, a_task_df,
        arriving_rate_df, weekday, proveng_dict, city_num_2_name, reduce_V
    )
    return final_revenue, final_allocation_for_a_day


# def get_revenue_by_task_and_num(task_lv, num):
#     revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
#     return revenue_for_lv[task_lv - 1] * num
def rnd_allocate_servers_2_citys(remain_servers_df, a_task_df, a_city_distance_df):
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    allocation_data = []
    cities_remain = a_task_df.index.tolist()
    max_level = int(a_task_df.columns.max()[6:])
    final_allocation_for_a_day = []
    final_revenue = 0
    for server in remain_servers_df.index:
        if len(cities_remain) == 0:
            break
        server_id = int(server[7:])
        server_level = remain_servers_df.loc[server, "lv"]
        server_city = remain_servers_df.loc[server, "current_city"]
        available_cities = [
            (city, level)
            for city in cities_remain
            for level in range(server_level, max_level + 1)
            if a_task_df.loc[city, f"level {level}"] == 1
        ]
        if len(available_cities) == 0:
            continue

        # print(f"{server_level=} {available_cities=}")
        city = random.choice(available_cities)
        task_level = city[1]
        task_city = int(city[0][5:])
        revenue = revenue_for_lv[task_level - 1]
        cost = a_city_distance_df[server_city][task_city]

        final_revenue += revenue - cost
        final_allocation_for_a_day.append(
            [server_id, server_city, task_city, task_level, server_level]
        )
        # print(f"Server: {server}, Level: {server_level}, City: {city}, City Level: {city[1]}")
        # ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        allocation_data.append({"server": server, "city": city})
        cities_remain.remove(city[0])

    allocation_df = pd.DataFrame(allocation_data, columns=["server", "city"])

    return final_revenue, final_allocation_for_a_day


def nearest_distance_allocate_servers_2_citys(
    remain_servers_df, a_task_df, a_city_distance_df
):
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    allocation_data = []
    cities_remain = a_task_df.index.tolist()
    max_level = int(a_task_df.columns.max()[6:])
    final_allocation_for_a_day = []
    final_revenue = 0
    for server in remain_servers_df.index:
        if len(cities_remain) == 0:
            break

        server_id = int(server[7:])
        server_level = remain_servers_df.loc[server, "lv"]
        server_city = remain_servers_df.loc[server, "current_city"]
        available_cities = [
            (city, level)
            for city in cities_remain
            for level in range(server_level, max_level + 1)
            if a_task_df.loc[city, f"level {level}"] == 1
        ]
        if len(available_cities) == 0:
            continue

        # print(f"{server_level=} {available_cities=}")
        # city = find_nearest_distance()

        min_dist = float("inf")
        min_city = None
        for eval_city in available_cities:
            eval_city_id = int(eval_city[0][5:])
            if a_city_distance_df[server_city][eval_city_id] < min_dist:
                min_city = eval_city
                min_dist = a_city_distance_df[server_city][eval_city_id]

        city = min_city
        task_level = city[1]
        task_city = int(city[0][5:])
        revenue = revenue_for_lv[task_level - 1]
        cost = a_city_distance_df[server_city][task_city]

        final_revenue += revenue - cost
        final_allocation_for_a_day.append(
            [server_id, server_city, task_city, task_level, server_level]
        )
        # print(f"Server: {server}, Level: {server_level}, City: {city}, City Level: {city[1]}")
        # ['业务员编号id', '业务员城市', '分配去的城市编号', '城市等级', '业务员等级']
        allocation_data.append({"server": server, "city": city})
        cities_remain.remove(city[0])

    allocation_df = pd.DataFrame(allocation_data, columns=["server", "city"])

    return final_revenue, final_allocation_for_a_day


def single_stage_opt_allocate_servers_2_citys(
    remain_servers_df, a_task_df, a_city_distance_df
):
    # 根据 lv 来做分组, join 分组后的结果应该
    join_idx_2_task_lv_server_lv_num = generate_idx_2_joins(
        a_task_df, remain_servers_df
    )
    # print(f"{join_idx_2_task_lv_server_lv_num=}")
    cnt = 0
    # print("（业务员编号id，业务员城市，分配去的城市编号，城市等级，业务员等级）")

    join_idx_2_revenue_and_combination_list_dict = {}
    revenue_sum = 0
    # 对于每一个集合，进行集合内分配
    for (
        join_idx,
        decisions_for_each_joins,
    ) in join_idx_2_task_lv_server_lv_num.items():
        cnt += 1
        # 根据集合分组后的，集合内level的数量分配
        # print("----join_idx, decisions_for_each_join----", join_idx, decisions_for_each_joins)
        # decisions_for_each_join = [[(1, 1, 4)]]
        revenue_and_combination_list_for_a_join = {}

        for idx, a_decision in enumerate(decisions_for_each_joins):
            # 集合分组后，level的分配方式可能很多, 一个decision里面是一个分配方案[(1, 1, 4)]
            # 可能是这样的 [(1, 1, 4)] / [(1, 1, 2), (2, 2, 2)]
            # 根据数量分配，进行实际的城市分配方案 list下的一个list是
            # print(f"{idx=} {a_decision=}")
            revenue_and_combination_for_decision = allocate_servers_2_cities(
                a_decision, a_task_df, remain_servers_df, a_city_distance_df
            )
            # revenue_and_combination_for_decision = {'revenue':final_revenue, 'combination':min_cost_city_id_of_server_and_task_combination}
            revenue_and_combination_list_for_a_join.update(
                {idx: revenue_and_combination_for_decision}
            )

        join_idx_2_revenue_and_combination_list_dict.update(
            {join_idx: revenue_and_combination_list_for_a_join}
        )

    # 有了不同的，开始合并决策，并reduce 并获取V历史记录的对应的收益。
    final_revenue_n_combination_list = get_all_allocations_for_decisions(
        join_idx_2_revenue_and_combination_list_dict
    )

    # 获取最好的分配
    final_revenue, final_allocation_for_a_day = get_allocation_for_a_day(
        final_revenue_n_combination_list,
        remain_servers_df, a_task_df,
        # arriving_rate_df, weekday, proveng_dict, city_num_2_name,reduce_V #TODO 
    )
    return final_revenue, final_allocation_for_a_day


def save_a_state_revenue(
    a_servers_df,
    new_servers_df,
    a_task_df,
    allocate_task_df,
    final_revenue,
    reduce_V,
    weekday,
    proveng_dict,
    city_num_2_name,
):
    reduced_server = reduce_server_df(
        a_servers_df=a_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_server_allocated = reduce_server_df(
        a_servers_df=new_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_task = reduce_task_df(
        a_task_df=a_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_allocate_task = reduce_task_df(
        a_task_df=allocate_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )

    reduce_V[weekday - 1].update(
        {
            (
                str(reduced_server),
                str(reduced_server_allocated),
                str(reduced_task.values.tolist()),
                str(reduced_allocate_task.values.tolist()),
            ): final_revenue
        }
    )

    return reduce_V


def get_a_state_revenue(
    a_servers_df,
    new_servers_df,
    a_task_df,
    allocate_task_df,
    reduce_V,
    weekday,
    proveng_dict,
    city_num_2_name,
):
    reduced_server = reduce_server_df(
        a_servers_df=a_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_server_allocated = reduce_server_df(
        a_servers_df=new_servers_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_task = reduce_task_df(
        a_task_df=a_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )
    reduced_allocate_task = reduce_task_df(
        a_task_df=allocate_task_df,
        proveng_dict=proveng_dict,
        city_num_2_name=city_num_2_name,
    )

    revenue = reduce_V[weekday - 1][(
                str(reduced_server),
                str(reduced_server_allocated),
                str(reduced_task.values.tolist()),
                str(reduced_allocate_task.values.tolist()),
            )]
    

    return revenue


def save_reduct_v(a_task_df, a_servers_df, final_allocation_for_a_day, arriving_rate_df, weekday, proveng_dict, city_num_2_name, final_revenue, reduce_V):
    new_task_df = a_task_df.copy()  # 初始
    allocate_task_df = allcocate_comb_2_allocate_task_df(
        final_allocation_for_a_day, new_task_df
    )
    final_revenue = final_revenue
    new_task_df = new_task_df - allocate_task_df  # 分配

    current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
    # 生成新任务
    arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
    new_tasks = generate_tasks(arriving_rate_matrix)

    # 更新状态矩阵
    new_task_df = update_state(current_task_df, new_tasks)

    # 更新业务员矩阵
    new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

    reduce_V = save_a_state_revenue(
        a_servers_df,
        new_servers_df,
        a_task_df,
        allocate_task_df,
        final_revenue,
        reduce_V,
        weekday,
        proveng_dict,
        city_num_2_name,
    )

    return reduce_V, new_task_df, new_servers_df, allocate_task_df



def cul_a_cycle(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")

        # remain_servers_df
        final_revenue, final_allocation_for_a_day = allocate_servers_2_citys_MDP(
            remain_servers_df, a_task_df, a_city_distance_df,
            arriving_rate_df, weekday, proveng_dict, city_num_2_name,reduce_V
        )

        reduce_V, new_task_df, new_servers_df, allocate_task_df = save_reduct_v(a_task_df, a_servers_df, final_allocation_for_a_day, arriving_rate_df, weekday, proveng_dict, city_num_2_name, final_revenue, reduce_V)

        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


def cul_a_cycle_rnd(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        final_revenue, final_allocation_for_a_day = rnd_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        new_task_df = a_task_df.copy()  # 初始
        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, new_task_df
        )

        new_task_df = new_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


def cul_a_cycle_nearest(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        (
            final_revenue,
            final_allocation_for_a_day,
        ) = nearest_distance_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        new_task_df = a_task_df.copy()  # 初始
        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, new_task_df
        )

        new_task_df = new_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


def cul_a_cycle_single_stage(
    T,
    a_servers_df,
    a_task_df,
    arriving_rate_df,
    a_city_distance_df,
    proveng_dict,
    city_num_2_name,
    reduce_V,
):
    # 每次迭代，应该决策前的状态，最优决策，缩减的决策和状态，以及收益， 之后应当可以通过缩减的决策和状态获取收益，这个得做一个保存
    # 我们先假设组内直接求最优，组间组合的时候做一个缩减后的状态收益决策。
    for weekday in range(1, T + 1):
        # ------根据 lv 来做分组，进行组内分配
        print(
            f"-----------------------------------------------weekday:{weekday}------------------------------------------"
        )
        # 先排除当天放假的员工
        remain_servers_df = a_servers_df[a_servers_df["day off"] != weekday]
        # print(f"{remain_servers_df=}")
        (
            final_revenue,
            final_allocation_for_a_day,
        ) = single_stage_opt_allocate_servers_2_citys(
            remain_servers_df, a_task_df, a_city_distance_df
        )

        new_task_df = a_task_df.copy()  # 初始
        allocate_task_df = allcocate_comb_2_allocate_task_df(
            final_allocation_for_a_day, new_task_df
        )

        new_task_df = new_task_df - allocate_task_df  # 分配
        arriving_rate_matrix = arriving_rate_df.values  # 将到达率转换为NumPy数组
        current_task_df = new_task_df  # 根据你的实际情况创建当前状态的DataFrame
        # 生成新任务
        new_tasks = generate_tasks(arriving_rate_matrix)

        # 更新状态矩阵
        new_task_df = update_state(current_task_df, new_tasks)

        # 更新业务员矩阵
        new_servers_df = update_server_cities(a_servers_df, final_allocation_for_a_day)

        reduced_server = reduce_server_df(
            a_servers_df=a_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_server_allocated = reduce_server_df(
            a_servers_df=new_servers_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_task = reduce_task_df(
            a_task_df=a_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )
        reduced_allocate_task = reduce_task_df(
            a_task_df=allocate_task_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
        )

        reduce_V[weekday - 1].update(
            {
                (
                    str(reduced_server),
                    str(reduced_server_allocated),
                    str(reduced_task.values.tolist()),
                    str(reduced_allocate_task.values.tolist()),
                ): final_revenue
            }
        )
        a_task_df = new_task_df
        a_servers_df = new_servers_df

    return reduce_V


# 先判断是否可以分配，划分分配的区间，区间内任意分配，
def generate_joins(
    task_lv_count: pd.core.series.Series, server_lv_count: pd.core.series.Series
) -> list[list]:
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
    task_lv_num = task_lv_count.shape[0]  # 任务等级数量
    task_join_count = 0
    server_join_count = 0

    for idx, (task_count, server_count) in enumerate(
        zip(task_lv_count, server_lv_count)
    ):
        task_join_count += task_count
        server_join_count += server_count
        lv = idx + 1
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


def form_all_decision_list_by_tuple(
    decision_task_done_by_server_level: tuple, tasks, servers, levels
) -> list[tuple]:
    # （任务等级，业务员等级，数量）
    last_task_idx = 0
    decision_list_tuple = []
    for idx, task_num in enumerate(tasks):
        task_level = levels[idx]
        task_num = tasks[idx]
        level_count_for_a_level_task = Counter(
            decision_task_done_by_server_level[last_task_idx : last_task_idx + task_num]
        )
        decision_list_tuple.extend(
            [
                (task_level, num, count)
                for num, count in level_count_for_a_level_task.items()
            ]
        )
        last_task_idx += task_num
    return decision_list_tuple


def decisions_2_decisions_list_tuple(
    decisions, tasks, servers, levels
) -> list[list[tuple]]:
    decisions_list_tuple = []

    for decision in decisions:
        decisions_list_tuple.append(
            form_all_decision_list_by_tuple(decision, tasks, servers, levels)
        )

    return decisions_list_tuple


def remove_duplicates(list_of_lists_of_tuples):
    """去除 list 中重复的 list of tuples。

    Args:
      list_of_lists_of_tuples: 一个包含 list of tuples 的 list。

    Returns:
      一个不包含重复 list of tuples 的 list。
    """

    # 将 list of tuples 转换为 tuple of tuples。
    list_of_tuples_of_tuples = [
        tuple(list_of_tuples) for list_of_tuples in list_of_lists_of_tuples
    ]

    # 使用 set 来去除重复的 tuple of tuples。
    set_of_tuples_of_tuples = set(list_of_tuples_of_tuples)

    # 将 tuple of tuples 转换为 list of tuples。
    list_of_lists_of_tuples = [
        list(tuple_of_tuples) for tuple_of_tuples in set_of_tuples_of_tuples
    ]

    # 返回不包含重复 list of tuples 的 list。
    return list_of_lists_of_tuples


def generate_idx_2_joins(state_df, remain_servers_df):
    """本函数用于,"""
    # 城市对应等级任务矩阵
    city_lv_matrix_df = state_df.iloc[:, :]
    # 任务计数，用于划分集合
    task_lv_count = city_lv_matrix_df.sum()
    # 业务员等级计数，用于划分集合
    server_lv_count = remain_servers_df["lv"].value_counts().sort_index()
    all_lv_values = range(1, 6)
    server_lv_count = server_lv_count.reindex(all_lv_values, fill_value=0)
    level_num = city_lv_matrix_df.shape[0]  # 任务等级数量
    city_num = city_lv_matrix_df.shape[0]  # 城市数量
    # 根据levels， servers，tasks区分组别
    servers = list(server_lv_count)
    tasks = task_lv_count.to_list()
    levels = list(range(1, len(tasks) + 1))
    print(
        f"{levels=}",
    )
    print(f"{tasks=} {sum(tasks)=}")
    print(f"{servers=} {sum(servers)=}")
    joins = generate_joins(task_lv_count, server_lv_count)
    print(f"根据不同level的task数量和server数量, 组内数量分配结果如下 {joins=}")
    # 组别内allocate
    join_idx_2_task_lv_server_lv_num = {}
    for join_idx, join in enumerate(joins):
        level_indices = [levels.index(lv) for lv in join]  # 获取join中每个level对应的索引
        a_tasks = [tasks[index] for index in level_indices]  # 获取与join中每个level相对应的任务数量
        a_servers = [
            servers[index] for index in level_indices
        ]  # 获取与join中每个level相对应的服务器数量
        a_lvs = [levels[index] for index in level_indices]
        decisions = allocate(tasks=a_tasks, servers=a_servers, levels=a_lvs)
        decisions_list_tuple = decisions_2_decisions_list_tuple(
            decisions=decisions, tasks=a_tasks, servers=a_servers, levels=a_lvs
        )

        join_idx_2_task_lv_server_lv_num.update(
            {join_idx: remove_duplicates(decisions_list_tuple)}
        )
    # 对每一个decision，对于不同地方和不同的人，进行排列组合分配
    # 获取每个城市情况
    # 每个server情况

    return join_idx_2_task_lv_server_lv_num


# 2024年1月21日 21点42分
def change_df_city_name_2_idx(cities: pd.DataFrame) -> (pd.DataFrame, dict):
    """改名字为idx，同时返回新df和idx2name"""
    city_names = cities.columns
    city_num = len(city_names)
    city_name_nums = list(range(1, city_num + 1))
    city_num_2_name = {key: value for key, value in zip(city_name_nums, city_names)}
    new_cities = cities.copy()
    new_cities.columns = new_cities.index = city_name_nums
    return new_cities, city_num_2_name


# 2024年1月21日 20点42分
def get_proveng_city_dist_mat_df(path="data\中国各城市空间权重矩阵(1).xlsx"):
    proveng_city_dist_mat_df = pd.read_excel(
        path,
        sheet_name="地理距离矩阵",
    )
    return proveng_city_dist_mat_df


def get_city_2_proveng_dict(proveng_city_dist_mat_df=None):
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_proveng_dict = (
        proveng_city_dist_mat_df[["cityeng", "proveng"]]
        .drop_duplicates()
        .set_index("cityeng")["proveng"]
        .to_dict()
    )
    return city_to_proveng_dict


def get_city_series_to_city_dict(proveng_city_dist_mat_df=None):
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()

    city_series_to_city_dict = (
        proveng_city_dist_mat_df[["cityeng", "cityseries"]]
        .drop_duplicates()
        .set_index("cityseries")["cityeng"]
        .to_dict()
    )
    return city_series_to_city_dict


def get_city_to_city_series_dict(proveng_city_dist_mat_df=None):
    if proveng_city_dist_mat_df is None:
        proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    city_to_city_series_dict = (
        proveng_city_dist_mat_df[["cityeng", "cityseries"]]
        .drop_duplicates()
        .set_index("cityeng")["cityseries"]
        .to_dict()
    )
    return city_to_city_series_dict


def get_dist_mat_from_xls_df(
    proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx
):
    distance_mat = proveng_city_dist_mat_df.iloc[:, distance_mat_start_idx:]
    distance_mat.index = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values
    distance_mat.columns = proveng_city_dist_mat_df.iloc[:, cityseries_idx].values

    return distance_mat


def get_select_city_df(
    city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat
):
    # 挑选出安徽、江苏、浙江和上海的省份及对应的矩阵数据
    select_proveng = ["Anhui", "Jiangsu", "Zhejiang", "Shanghai"]  # TODO 可能要换省份
    provengs = proveng_city_dist_mat_df.iloc[:, proveng_idx].values  # 表格第一列为城市名称
    select_proveng_idxs = [
        i for i, prov in enumerate(provengs) if prov in select_proveng
    ]  # xls里选取指定的省份的idx，是字母
    select_cityeng_idxs = proveng_city_dist_mat_df.values[
        select_proveng_idxs, cityseries_idx
    ]  # xls指定省份对应的城市id
    select_idx = pd.Index(select_cityeng_idxs)  # 创建对应索引
    # 使用 loc 方法选择相应的行和列
    select_proveng_city_df = distance_mat.loc[select_idx, select_idx]

    # 从城市列表中随机选择city_num-1个城市（不包括上海）
    rnd_cities = np.random.choice(
        select_cityeng_idxs[1:], city_num - 1, replace=False
    )  # 第一列为上海，所以从1开始
    # 将上海添加到随机选择的城市列表中 # City158 = shanghai
    select_cities = ["City158"] + rnd_cities.tolist()
    select_city_idx = pd.Index(select_cities)
    # 使用 loc 方法选择相应的行和列
    select_city_df = select_proveng_city_df.loc[select_city_idx, select_city_idx]

    return select_city_df


def generate_city(city_num: int = 26) -> (pd.DataFrame, dict):
    """本函数功能，生成指定数量城市的城市距离矩阵，必然包含上海。返回的dataframe包含城市坐标和省份坐标/和一个城市转省份的dict"""

    # 地理距离矩阵xls
    proveng_city_dist_mat_df = get_proveng_city_dist_mat_df()
    # 从第四列开始，是距离矩阵内容
    proveng_idx = 0
    cityeng = 1
    cityseries_idx = 2
    distance_mat_start_idx = 3

    #
    city_series_to_city_dict = get_city_series_to_city_dict(
        proveng_city_dist_mat_df=proveng_city_dist_mat_df
    )
    city_to_proveng = get_city_2_proveng_dict(
        proveng_city_dist_mat_df=proveng_city_dist_mat_df
    )

    # 地理距离矩阵xls 第三列是城市id, city_*, 设定纵坐标
    distance_mat = get_dist_mat_from_xls_df(
        proveng_city_dist_mat_df, distance_mat_start_idx, cityseries_idx
    )
    select_city_df = get_select_city_df(
        city_num, proveng_city_dist_mat_df, proveng_idx, cityseries_idx, distance_mat
    )

    city_series_columns = select_city_df.columns
    city_columns = [
        (
            city_to_proveng[city_series_to_city_dict[item]],
            city_series_to_city_dict[item],
        )
        for item in city_series_columns
    ]  # 最为新矩阵的idx和col

    column_names = ["proveng", "cityeng"]  # 给idx的idx
    city_df = pd.DataFrame(
        select_city_df.values,
        index=pd.MultiIndex.from_tuples(city_columns, names=column_names),
        columns=pd.MultiIndex.from_tuples(city_columns, names=column_names),
    )
    return city_df


if __name__ == "__main__":
    city = generate_city(city_num=10)
    city_to_proveng = get_city_2_proveng_dict()
