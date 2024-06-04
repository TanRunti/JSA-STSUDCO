import math
import random

import numpy as np
from Usernum_record import user_name_set
from DAG_Builder import DagBuilder


#################### 构建整体卸载环境 ####################
class UAVEnv(object):
    height = ground_length = ground_width = 200
    sum_task_size = 100 * 1048576
    loc_uav = [100, 100]
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6
    p_noisy_los = 10 ** (-13)
    p_noisy_nlos = 10 ** (-11)
    flight_speed = 100
    f_ue = 1e9
    f_uav = 5e9
    r = 10 ** (-28)
    s = 1000
    p_uplink = 0.1
    p_uplink_cloud = 0.2
    alpha0 = 1e-5
    T = 320
    t_fly = 1
    t_com = 7
    delta_t = t_fly + t_com
    v_ue = 1
    slot_num = int(T / delta_t)
    m_uav = 9.65
    e_battery_uav = 1000000

    loc_cloud = [300, 300]
    f_cloud = 1e10
    p_cloud = 0.2

    #################### ues ####################
    obj = user_name_set()
    # User number
    M = 30
    # Nlock flag
    block_flag_list = np.random.randint(0, 2, M)
    # Location
    loc_ue_list = np.random.randint(0, 201, size=[M, 2])
    # Task data
    task_list = np.random.randint(1572864, 6291456, M)
    var = 0.01
    # Cropping
    action_bound = [-1, 1]
    # Ue id；direction；distance；offloading decision
    action_dim = 4
    # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
    state_dim = 4 + M * 4

    # Set instance number
    instance_tag = 0
    # Set instance number
    instance_num = [6, 12, 18, 24]
    # Calculate group number
    group_j = int(M / instance_num[instance_tag])
    # Calculate intermediate layer
    inter_layer_nums = int((M - 2 * group_j) / 2)

    task_complete = [1 for _ in range(M + 1)]
    task_complete[0] = 0
    dag = DagBuilder()
    task_depend = dag.return_dag()

    # print(task_depend)
    # Initialization
    def __init__(self):
        self.total_reward = 0
        # uav battery remain, uav loc,
        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        # remaining sum task size,
        self.start_state = np.append(self.start_state, self.sum_task_size)
        # all ue loc,
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        # all ue task size,
        self.start_state = np.append(self.start_state, self.task_list)
        # all ue block_flag
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    # Reset environment
    def reset_env(self):
        obj = user_name_set()
        M = obj.get_user_num()
        self.sum_task_size = 100 * 1048576
        self.e_battery_uav = 1000000
        self.loc_uav = [100, 100]

        self.task_complete = [1 for _ in range(M + 1)]
        self.task_complete[0] = 0
        self.loc_ue_list = np.random.randint(0, 201, size=[M, 2])
        self.total_reward = 0
        self.reset_step()

    # Reset step
    def reset_step(self):
        obj = user_name_set()
        M = obj.get_user_num()
        self.task_list = np.random.randint(1572864, 6291456, M)
        self.block_flag_list = np.random.randint(0, 2, M)

    # Reset
    def reset(self):
        self.reset_env()
        # uav battery remain, uav loc, remain sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    # Get current observation
    def _get_obs(self):
        # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, self.sum_task_size)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def reset2(self, delay, x, y, into_cloud, task_size, ue_id, dis, chose_label, t, e):
        obj = user_name_set()
        M = obj.get_user_num()
        self.sum_task_size -= self.task_list[ue_id]
        self.total_reward += -delay
        self.task_complete[ue_id] = 0
        for i in range(M):
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2
            dis_ue = tmp[1] * self.delta_t * self.v_ue
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self.reset_step()

    def step(self, action):
        # action type：0: ue id ; 1: direction; 2: distance; 3. into cloud
        obj = user_name_set()
        M = obj.get_user_num()
        step_redo = False
        is_terminal = False
        reset_dist = False
        # Avoid action boundaries
        action = (action + 1) / 2
        if action[0] == 1:
            ue_id = M - 1
        else:
            ue_id = int(M * action[0])

        parent = 0
        for i_index in range(int(self.inter_layer_nums / self.group_j)):
            parent += self.task_complete[self.task_depend[ue_id + 1][i_index] + 1]

        theta = action[1] * np.pi * 2
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

        if action[3] > 0.5:
            into_cloud = True
        else:
            into_cloud = False

        ######################
        # Distance
        dis_fly = action[2] * self.flight_speed * self.t_fly
        # Flight energy consumption
        e_fly = (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5
        # UAV location
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav
        # Distance between the UAV and the ue
        trans_dis = math.sqrt((self.loc_ue_list[ue_id][0] - loc_uav_after_fly_x)**2 + (self.loc_ue_list[ue_id][1] -
                                                                                       loc_uav_after_fly_y)**2)

        ######################
        # The time delay and energy consumption of the local computing
        t_local = self.com_delay_local(task_size)

        e_local = self.com_energy_local(task_size)

        ######################
        # The time delay and energy consumption of the UAV-edge computing
        t_edge_trans = self.tran_delay_uav(self.loc_ue_list[ue_id], np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                                       task_size, block_flag)
        t_edge_com = self.com_delay_uav(task_size)
        t_edge = t_edge_trans + t_edge_com

        e_edge_trans = self.tran_energy_uav(t_edge_trans)
        e_edge_com = self.com_energy_uav(task_size)
        e_edge = e_edge_trans + e_edge_com

        ######################
        # The time delay and energy consumption of the cloud computing
        t_cloud_trans = self.tran_delay_cloud(np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]), task_size)
        t_cloud = t_edge_trans + t_cloud_trans + self.com_delay_cloud(task_size)

        e_cloud_trans = self.tran_energy_cloud(t_cloud_trans)
        e_cloud = e_edge_trans + e_cloud_trans + self.com_energy_cloud(task_size)

        if into_cloud:
            weighted_sum_min = 0.5 * t_cloud + 0.5 * e_cloud
        else:
            weighted_sum_min = min([0.5 * t_local + 0.5 * e_local,
                                    0.5 * t_edge + 0.5 * e_edge])

        # Decision type label
        chose_label = 0
        t = t_local
        e = e_local
        if weighted_sum_min == 0.5 * t_edge + 0.5 * e_edge:
            chose_label = 1
            t = t_edge
            e = e_edge
        elif weighted_sum_min == 0.5 * t_cloud + 0.5 * e_cloud:
            chose_label = 2
            t = t_cloud
            e = e_cloud

        if weighted_sum_min > 0.5 * t_cloud + 0.5 * e_cloud:
            into_cloud_change = True

        ######################
        # Calculation situation
        # End
        if self.sum_task_size == 0:
            is_terminal = True
            reward = 0

        # Task size out of bounds
        elif self.sum_task_size - self.task_list[ue_id] < 0:
            step_redo = True
            self.task_list = np.ones(M) * self.sum_task_size
            reward = 0

        # Dependency error
        elif parent > 0:
            # 如果超出边界，则飞行距离dist置零
            delay = weighted_sum_min + e_fly/1000
            reward = (-1) * parent - delay
            if chose_label == 1:
                self.e_battery_uav = self.e_battery_uav - e_edge_com
            elif chose_label == 2:
                self.e_battery_uav = self.e_battery_uav - e_cloud_trans
            self.loc_uav[0] = loc_uav_after_fly_x
            self.loc_uav[1] = loc_uav_after_fly_y
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, into_cloud, task_size,
                        ue_id, trans_dis, chose_label, t, e)

        # UAV location error
        elif loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width or loc_uav_after_fly_y < 0 or \
                loc_uav_after_fly_y > self.ground_length:
            reset_dist = True
            # delay = weighted_sum_min
            # delay = weighted_sum_min + e_fly/2000 + trans_dis/100
            # delay = weighted_sum_min + trans_dis/100
            delay = weighted_sum_min + e_fly / 1000
            reward = -delay
            if chose_label == 1:
                self.e_battery_uav = self.e_battery_uav - e_edge_com
            elif chose_label == 2:
                self.e_battery_uav = self.e_battery_uav - e_cloud_trans
            self.reset2(delay, self.loc_uav[0], self.loc_uav[1], into_cloud, task_size, ue_id, trans_dis,
                        chose_label, t, e)

        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_edge_com or \
                self.e_battery_uav - e_fly < e_cloud_trans:
            delay = 0.5 * t_local + 0.5 * e_local
            reward = -delay
            into_cloud = 0
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, into_cloud, task_size, ue_id, is_terminal,
                        chose_label, t, e)

        else:
            delay = weighted_sum_min + e_fly/1000
            reward = -delay
            if chose_label == 1:
                self.e_battery_uav = self.e_battery_uav - e_edge_com
            elif chose_label == 2:
                self.e_battery_uav = self.e_battery_uav - e_cloud_trans
            self.loc_uav[0] = loc_uav_after_fly_x
            self.loc_uav[1] = loc_uav_after_fly_y
            self.reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y, into_cloud, task_size,
                        ue_id, trans_dis, chose_label, t, e)

        return self._get_obs(), reward, is_terminal, step_redo, reset_dist, into_cloud, loc_uav_after_fly_x, \
            loc_uav_after_fly_y

    ###########################
    ####### Delay
    ###########################
    def com_delay_local(self, task_size):
        t_local_com = task_size / (self.f_ue / self.s)

        return t_local_com

    def tran_delay_uav(self, loc_ue, loc_uav, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)
        t_trans_uav = task_size / trans_rate

        return t_trans_uav

    def com_delay_uav(self, task_size):
        t_edge_com = task_size / (self.f_uav / self.s)

        return t_edge_com

    def tran_delay_cloud(self, loc_uav, task_size):
        dx = 250 - loc_uav[0]
        dy = 250 - loc_uav[1]
        dh = self.height
        dist_uav_cloud = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        g_uav_cloud = abs(self.alpha0 / dist_uav_cloud ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink_cloud * g_uav_cloud / p_noise)
        t_trans_cloud = task_size / trans_rate

        return t_trans_cloud

    def com_delay_cloud(self, task_size):
        t_cloud_com = task_size / (self.f_cloud / self.s)

        return t_cloud_com

    ###########################
    ####### Energy consumption
    ###########################
    def com_energy_local(self, task_size):
        e_local_com = self.r * self.f_ue ** 2 * task_size * self.s

        return e_local_com

    def tran_energy_uav(self, t_trans_uav):
        e_trans_uav = self.p_uplink * t_trans_uav

        return e_trans_uav

    def com_energy_uav(self, task_size):
        e_edge_com = self.p_uplink_cloud * task_size * self.s / self.f_uav

        return e_edge_com

    def tran_energy_cloud(self, t_trans_cloud):
        e_trans_cloud = self.p_uplink_cloud * t_trans_cloud

        return e_trans_cloud

    def com_energy_cloud(self, task_size):
        e_cloud_com = self.p_cloud * task_size * self.s / self.f_cloud

        return e_cloud_com