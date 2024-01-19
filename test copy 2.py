import numpy as np
import mdptoolbox

# 定义城市和业务员的信息
cities = ['City A', 'City B', 'City C']  # 城市列表
num_tasks = [10, 5, 8]  # 每个城市的任务数量
task_levels = [1, 2, 3, 4, 5]  # 任务等级
agents = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']  # 业务员列表
agent_levels = [1, 2, 3, 4]  # 业务员等级

# 定义奖励和成本参数
revenue_for_lv = [3500, 3000, 2500, 2000, 1500]  # 不同任务等级的奖励
cost_matrix = np.array([[0, 1, 3, 5], [1, 0, 2, 4], [3, 2, 0, 1]])  # 城市间的移动成本（代价）

# 定义状态空间和动作空间
states = [(c, t, a) for c in cities for t in task_levels for a in agents]
actions = agents

# 定义奖励函数
rewards = np.zeros((len(states), len(actions)))  # 初始化奖励矩阵

for i, state in enumerate(states):
    city, task_level, _ = state
    reward = revenue_for_lv[task_level - 1]  # 根据任务等级获取奖励
    rewards[i, :] = reward

# 定义状态转移概率矩阵
transition_matrix = np.zeros((len(actions), len(states), len(states)))  # 初始化状态转移概率矩阵

for i, action in enumerate(actions):
    for j, state in enumerate(states):
        city, task_level, _ = state
        for k, next_state in enumerate(states):
            next_city, next_task_level, next_agent = next_state
            if next_city == city and next_agent == action:
                # 当下一个状态的城市和业务员与当前动作匹配时，计算转移概率
                transition_matrix[i, j, k] = 1.0 / num_tasks[cities.index(city)]

# 定义MDP模型
mdp_model = mdptoolbox.mdp.ValueIteration(transition_matrix, rewards, discount=0.9)
mdp_model.run()

# 打印最优值函数和最优策略
print("最优值函数:")
print(mdp_model.V)
print("\n最优策略:")
print(mdp_model.policy)