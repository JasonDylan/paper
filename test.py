import itertools

allocations_for_decisions = {
    1: [{'revenue': 11, 'combination': 'combination11'}, {'revenue':12, 'combination': 'combination12'}],
    2: [{'revenue':21, 'combination': 'combination21'}],
    3: [{'revenue':31, 'combination': 'combination31'}, {'revenue':32, 'combination': 'combination32'}, {'revenue':33, 'combination': 'combination33'}],
    4: [{'revenue':41, 'combination': 'combination41'}, {'revenue':42, 'combination': 'combination42'}],
}
print(allocations_for_decisions.values())
# Calculate the number of possible combinations
def get_all_allocations_for_decisions(allocations_for_decisions, ):

    num_combinations = 1
    for key, value in allocations_for_decisions.items():
        num_combinations *= len(value)

    # Print the number of possible combinations
    print(num_combinations)


    # Get all possible combinations of values from the dictionary
    combinations = list(itertools.product(*allocations_for_decisions.values()))

    # Print all combinations
    print(len(combinations))
    for idx,combination in enumerate(combinations):
        print(idx+1, combination)