# For arrayPropagation.py tests
'''
# Code for running methods above - feel free to uncomment
start = 3
num_turbines = 10


arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
effects_mark = 27

turbine_array_child(arr, nodeNames, start, num_turbines, effects_mark)
# turbine_array_parent(arr, nodeNames, start, num_turbines, effects_mark)

start = 12
arr, nodeNames = excel2Matrix("Task49Graph.xlsx", "AlteredSheet-noCause")
effects_mark = 49

turbine_array_child(arr, nodeNames, start, num_turbines, effects_mark)

start = 1
num_turbines = 10
update = False


arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
effects_mark = 27
turbine_array_parent_prob(arr, nodeNames, start, num_turbines, effects_mark)'''


'''
arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_child(arr, nodeNames, start, num_turbines, effects_mark)

arr1, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_child_prob(arr1, nodeNames, start, num_turbines, effects_mark)

arr2, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_child_prob(arr2, nodeNames, start, num_turbines, effects_mark, update = True)

arr3, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_parent(arr3, nodeNames, start, num_turbines, effects_mark)

arr4, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_parent_prob(arr4, nodeNames, start, num_turbines, effects_mark)

arr5, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
turbine_array_parent_prob(arr5, nodeNames, start, num_turbines, effects_mark, update = True)'''

'''start = [17]
num_turbines = 10
update = False
arr, nodeNames = excel2Matrix("failureData.xlsx", "bigMatrix")
C, D = monte_carlo_sim_array(1000000, 10, plot=True, start=start, adjacency_matrix=arr, nodeNames=nodeNames, rand_seed = True, mid_point=False)
print(D)'''