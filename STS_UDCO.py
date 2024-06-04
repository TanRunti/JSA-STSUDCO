import os
import sys
import time
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from Environment import UAVEnv
from Observation_Normal import ObserNormalization
from Usernum_record import user_name_set
from DAG_Builder import DagBuilder


class Config:
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class
        self.env_class = env_class
        self.env_args = env_args

        if env_args is None:
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']
        self.state_dim = env_args['state_dim']
        self.action_dim = env_args['action_dim']
        self.if_discrete = env_args['if_discrete']

        '''Network training independent variables'''
        self.gamma = 0.98
        self.net_dims = (256, 64)
        self.learning_rate = 6e-4  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(64)
        self.horizon_len = int(8)
        self.buffer_size = int(10000)
        self.repeat_times = 1.0

        '''Network evaluation independent variable'''
        self.cwd = None
        self.break_step = +np.inf
        self.eval_times = int(8)
        self.eval_per_step = int(8)

    def init_before_training(self):
        if self.cwd is None:
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)


class ActorSAC(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_s = build_mlp(dims=[state_dim, *dims])
        self.dec_a_avg = build_mlp(dims=[dims[-1], action_dim])
        self.dec_a_std = build_mlp(dims=[dims[-1], action_dim])
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state: Tensor) -> Tensor:
        state_tmp = self.enc_s(state)
        return self.dec_a_avg(state_tmp).tanh()

    def get_action(self, state: Tensor) -> Tensor:
        state_tmp = self.enc_s(state)
        action_avg = self.dec_a_avg(state_tmp)
        action_std = self.dec_a_std(state_tmp).clamp(-1.0, 1.0).exp()

        noise = torch.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.clip(-1.0, 1.0)

    def get_action_logprob(self, state: Tensor) -> [Tensor, Tensor]:
        state_tmp = self.enc_s(state)
        action_log_std = self.dec_a_std(state_tmp).clamp(-1.0, 1.0)
        action_std = self.dec_a_std(state_tmp).clamp(-1.0, 1.0).exp()
        action_avg = self.dec_a_avg(state_tmp)

        '''Adding noise to random variables'''
        noise = torch.randn_like(action_avg, requires_grad=True)
        a_noise = action_avg + action_std * noise

        '''Calculate log based on the mean and standard deviation of the noise adding action'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = action_log_std + self.log_sqrt_2pi + noise.pow(2) * 0.5

        '''Fix log through orthogonal function'''
        log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)


# 构建一对评价网络
class CriticTwin(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_sa = build_mlp(dims=[state_dim + action_dim, *dims])
        self.dec_q1 = build_mlp(dims=[dims[-1], action_dim])
        self.dec_q2 = build_mlp(dims=[dims[-1], action_dim])

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        return self.dec_q1(sa_tmp)

    def get_q1_q2(self, state, action):
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        return self.dec_q1(sa_tmp), self.dec_q2(sa_tmp)


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]
    return nn.Sequential(*net_list)


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}


def build_env(env_args):
    env = UAVEnv()
    return env


class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.gamma = args.gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.soft_update_tau = args.soft_update_tau

        self.last_state = None
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), args.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), args.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau: float):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentSAC(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)

        self.alpha_log = torch.tensor(-1, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)
        self.obj_c = 1.0

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        is_terminals = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        now_state = self.last_state
        get_action = self.act.get_action
        s_normal = ObserNormalization()
        for i in range(horizon_len):
            state = torch.as_tensor(s_normal.obser_normal(now_state), dtype=torch.float32, device=self.device)
            # action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]
            action = get_action(state.unsqueeze(0))[0]
            ary_action = action.cpu().detach().numpy()
            if if_random:
                action_var = torch.randn_like(action) / 10.0
                action_var = action_var.cpu().detach().numpy()
                # action = torch.clamp(action + action_var, -1, 1)
                action = np.add(ary_action, action_var / 2)
                action = np.clip(action, -1, 1)

            action = torch.tensor(action)
            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            now_state, reward, is_terminal, step_redo, reset_dist, into_cloud, x, y = env.step(ary_action)

            rewards[i] = reward
            is_terminals[i] = is_terminal

            if step_redo:
                continue
            if reset_dist:
                action[2] = -1
            if into_cloud:
                action[3] = 1
            if is_terminal:
                now_state = s_normal.obser_normal(env.reset())

        self.last_state = now_state
        rewards = rewards.unsqueeze(1)
        un_terminals = (1.0 - is_terminals.type(torch.float32)).unsqueeze(1)

        return states, actions, rewards, un_terminals

    def update_net(self, buffer) -> [float]:
        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        update_a = 0
        '''Main contribution'''
        for update_c in range(1, update_times + 1):
            '''Update Critic'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            # Calculate tao
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()
            reliable_lambda = math.exp(-self.obj_c ** 2)
            # Adaptive adjustment of the relative frequency
            if update_a / update_c < 1 / (2 - reliable_lambda):
                '''Update temperature factor'''
                action, logprob = self.act.get_action_logprob(state)  # policy gradient
                obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
                self.optimizer_update(self.alpha_optim, obj_alpha)

                '''Update Actor'''
                alpha = self.alpha_log.exp()
                obj_actor = (self.cri(state, action) + logprob * alpha).mean()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, un_terminals, next_state = buffer.sample(batch_size)
            next_action, next_log_prob = self.act.get_action_logprob(next_state)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_state, next_action))
            alpha = self.alpha_log.exp()
            q_label = reward + self.gamma * (next_q + next_log_prob * alpha)

        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.

        return obj_critic, state

class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.un_terminals = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, un_terminals = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True

            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size
            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.un_terminals[p0:p1], self.un_terminals[0:p] = un_terminals[:p2], un_terminals[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.un_terminals[self.p:p] = un_terminals
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)  # indices
        return self.states[ids], self.actions[ids], self.rewards[ids], self.un_terminals[ids], self.states[ids + 1]
        # return self.states[ids], self.actions[ids], self.rewards[ids], self.states[ids + 1]


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int, eval_times: int = 2, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.recorder = []
        print("\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()
        std_r = rewards_steps_ary[:, 0].std()
        avg_s = rewards_steps_ary[:, 1].mean()

        # avg_r = avg_r * 2.0 / 3.0

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def get_rewards_and_steps(env, actor, is_terminal: bool = False) -> (float, int):
    device = next(actor.parameters()).device

    s_normal = ObserNormalization()
    state = s_normal.obser_normal(env.reset())
    episode_steps = 0
    cumulative_returns = 0.0
    for episode_steps in range(12345):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]
        state, reward, is_terminal, step_redo, reset_dist, into_cloud ,x ,y = env.step(action)
        cumulative_returns += reward

        # if if_render:
        #     env.render()
        if is_terminal:
            break
    return cumulative_returns, episode_steps + 1


def train_agent(args: Config):  # todo add gpu_id
    args.init_before_training()
    gpu_id = -1

    env = build_env(args.env_args)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.last_state = env.reset()
    # buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size, state_dim=args.state_dim,
    #                       action_dim=args.action_dim)
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size, state_dim=args.state_dim,
                          action_dim=args.action_dim)
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    # buffer_items = agent.explore_env(env, 80, if_random=True)

    buffer.update(buffer_items)

    evaluator = Evaluator(eval_env=env,
                          eval_per_step=args.eval_per_step, eval_times=args.eval_times, cwd=args.cwd)
    torch.set_grad_enabled(False)

    while True:
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)

        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break


def train_sac_for_UAVMEC():
    from Usernum_record import user_name_set
    obj = user_name_set()
    user_num = obj.get_user_num()
    # user_num = 50
    env_args = {
        'env_name': 'UAV-MEC',
        'state_dim': int(user_num) * 4 + 4,
        'action_dim': 4,
        'if_discrete': False
    }

    args = Config(agent_class=AgentSAC, env_class=UAVEnv, env_args=env_args)
    args.break_step = int(10000)
    args.net_dims = (256, 64)
    args.gpu_id = -1
    args.gamma = 0.98

    train_agent(args)


user_nums = 30
obj = user_name_set()
obj.receive_user_num(user_nums)
dag = DagBuilder()
task_depend = dag.return_dag()
train_sac_for_UAVMEC()
