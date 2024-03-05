import pandas as pd
import numpy as np
from city_cls import (
    generate_city,
    get_city_2_proveng_dict,
    change_df_city_name_2_idx,
    cul_a_cycle,
)
import random

global revenue_for_lv, a_city_distance_df, province_dict, city_num_2_name

if __name__ == "__main__":
    # 全局的一些设定
    ## 收益率
    revenue_for_lv = [3500, 3000, 2500, 2000, 1500]
    # 全局设定种子，保证，每次随机结果一致
    np.random.seed(42)
    # 生成城市规模/一个省
    city_num = 26
    # 生成 26个城市
    city_distance_df = generate_city(city_num=city_num)
    city_to_proveng = get_city_2_proveng_dict()

    proveng = city_distance_df.index.get_level_values(
        "proveng"
    ).unique()  # 获得所有province的name
    proveng_dict = {proveng[i]: i + 1 for i in range(len(proveng))}
    city_names = city_distance_df.columns
    a_city_distance_df, city_num_2_name = change_df_city_name_2_idx(
        cities=city_distance_df
    )

    global arriving_rate_df
    
    travel_fee_df = pd.read_excel(
        "./data/数据.xlsx", sheet_name="travel fee", index_col=0
    )
    arriving_rate_df = pd.read_excel(
        "./data/数据.xlsx", sheet_name="arriving rate", index_col=0
    )
    initial_state_df = pd.read_excel(
        "./data/数据.xlsx", sheet_name="initial state", index_col=0
    )
    servers_df = pd.read_excel("./data/数据.xlsx", sheet_name="servers", index_col=0)
    # 员工
    servers_df.columns = ["current_city", "lv", "day off"]

    # 对于每个join，产生其分配方案，生成所有分配方案，分配方案是指
    # 当前日子，对于每个城市的状态，业务员的状态，生成一组对业务员的分配
    # 可能为（业务员编号id，业务员城市，分配去的城市编号，业务员等级，城市等级

    arriving_rate_df = arriving_rate_df[:city_num]
    a_state_df = initial_state_df.copy()[:city_num]
    a_servers_df = servers_df.copy()[servers_df["current_city"] <= city_num]
    saved_params = {}
    T = 7
    reduce_V = [{} for _ in range(T)]
    # 输入一个缩减为三个省的state 和 一个action 对应结果为 对应的收益
    # 这个循环，进行一次指定周期内的迭代。
    reduce_V_actual = [{} for _ in range(T)]
    random.seed(42)
    iters = 10000
    
    reduce_V_iter = [[{} for _ in range(T)] for a_iter in range(iters)]
    reduce_V_actual_iter = [[{} for _ in range(T)] for a_iter in range(iters)]
    for it in range(iters):
        reduce_V, reduce_V_iter = cul_a_cycle(
            T=T,
            a_servers_df=a_servers_df,
            a_task_df=a_state_df,
            arriving_rate_df=arriving_rate_df,
            a_city_distance_df=a_city_distance_df,
            proveng_dict=proveng_dict,
            city_num_2_name=city_num_2_name,
            reduce_V=reduce_V,
            reduce_V_iter=reduce_V_iter,
            reduce_V_actual=reduce_V_actual,
            a_iter = it
        )
        print(f"---------------------------------------------------------------iter:{it=}--------------------------------------------------------------- ")

        ## 计算V
        
        all_V_from_a_round =  reduce_V_iter[it]
        temp_sum_from_T = 0
        sum_diff = 0
        for weekday in range(T-1,-1,-1):
            kv = all_V_from_a_round[weekday]
            if len(kv.keys()) > 1:
                print(kv)
            key = list(kv.keys())[0]
            value = list(kv.values())[0]
            # print(f"{value}")
            temp_sum_from_T += value
            old_V_actual = reduce_V_actual[weekday].get(key, 0)
            sum_diff += abs(old_V_actual-temp_sum_from_T)
            print(f"{old_V_actual=}, {temp_sum_from_T=}")
            reduce_V_actual[weekday].update({key:temp_sum_from_T})
            reduce_V_actual_iter[it][weekday].update({key:temp_sum_from_T})
            # reduce_V_actual[it][] = 
        ## 上一次迭代和本地迭代结果差距计算, 问题：结果差如何计算，整个V矩阵计算么，还是对于单独一次的上下次计算？
        print(f"{sum_diff=}")
        if sum_diff< 1:
            break
    print(f"---------------------------------------------------------------DONE--------------------------------------------------------------- ")
