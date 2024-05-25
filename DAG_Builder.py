import math
import random

import numpy as np
from Usernum_record import user_name_set


# Build and parse DAG structure
class DagBuilder(object):
    def __init__(self):
        # User number
        M = 30
        # User list
        user_list = [i_user for i_user in range(M)]
        # Set instance number
        instance_tag = 0
        # Set instance number
        instance_num = [6, 12, 18, 24]
        # Calculate group number
        group_j = int(M / instance_num[instance_tag])
        # Calculate the number of the intermediate layer
        inter_layer_nums = int((M - 2 * group_j) / 2)
        # Task location list
        task_loc_list = np.zeros((1, M), dtype=int)
        task_start_list = np.zeros((1, group_j), dtype=int)
        task_second_list = np.zeros((1, inter_layer_nums), dtype=int)
        task_third_list = np.zeros((1, inter_layer_nums), dtype=int)
        task_end_list = np.zeros((1, group_j), dtype=int)
        # task dependency
        self.task_depend = np.zeros((M + 1, int(inter_layer_nums / group_j)), dtype=int)

        # Randomly generate DAG task order
        for i_start_end in range(group_j):
            start_tag = random.choice(user_list)
            task_loc_list[0][start_tag] = 1
            user_list.remove(start_tag)
            end_tag = random.choice(user_list)
            task_loc_list[0][end_tag] = 4
            user_list.remove(end_tag)

        for i_scd_trd in range(inter_layer_nums):
            second_tag = random.choice(user_list)
            task_loc_list[0][second_tag] = 2
            user_list.remove(second_tag)
            third_tag = random.choice(user_list)
            task_loc_list[0][third_tag] = 3
            user_list.remove(third_tag)

        # Group tasks
        a = b = c = d = 0
        for i_group in range(M):
            if task_loc_list[0][i_group] == 1:
                task_start_list[0][a] = i_group
                a += 1
            if task_loc_list[0][i_group] == 2:
                task_second_list[0][b] = i_group
                b += 1
            if task_loc_list[0][i_group] == 3:
                task_third_list[0][c] = i_group
                c += 1
            if task_loc_list[0][i_group] == 4:
                task_end_list[0][d] = i_group
                d += 1
        np.random.shuffle(task_start_list[0])
        np.random.shuffle(task_second_list[0])
        np.random.shuffle(task_third_list[0])
        np.random.shuffle(task_end_list[0])

        # Group DAGs
        # the start layer        (1)
        # the second layer    (2)   (3)
        # the third layer     (4)   (5)
        # the end layer          (6)
        for i_depend in range(group_j):
            self.task_depend[task_start_list[0][i_depend] + 1] = [0] * int(inter_layer_nums / group_j)

        for i_depend in range(inter_layer_nums):
            index, _ = divmod(i_depend, int(inter_layer_nums / group_j))
            self.task_depend[task_second_list[0][i_depend] + 1][0] = task_start_list[0][index]

        for i_depend in range(inter_layer_nums):
            index, _ = divmod(i_depend, int(inter_layer_nums / group_j))
            for i_index in range(int(inter_layer_nums / group_j)):
                self.task_depend[task_third_list[0][i_depend] + 1][i_index] = task_second_list[0][
                    i_index + index * int(inter_layer_nums / group_j)]

        for i_depend in range(group_j):
            for i_index in range(int(inter_layer_nums / group_j)):
                self.task_depend[task_end_list[0][i_depend] + 1][i_index] = task_third_list[0][
                    i_index + i_depend * int(inter_layer_nums / group_j)]

    def return_dag(self):
        return self.task_depend