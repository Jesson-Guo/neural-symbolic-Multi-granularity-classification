import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy

# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100

env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""
    def __init__(self, batch_size, num_states, num_actions, memory_capacity, episilon, lr, gamma, update_freq, device):
        super(DQN, self).__init__()
        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.episilon = episilon
        self.lr = lr
        self.gamma = gamma
        self.update_freq = update_freq
        self.device = device

        self.training = False

        self.eval_net = Net(self.num_states, self.num_actions).to(self.device)
        self.target_net = Net(self.num_states, self.num_actions).to(self.device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((self.memory_capacity, self.num_states * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def epsilon_calc(self, step, ep_min=0.01, ep_max=1, eps_total = 1000):
        # 动态eps
        return max(ep_min, ep_max - (ep_max - ep_min) * step / eps_total)

    def select_action(self, state):
        n = state.shape[0]
        # state = state.to(self.device).unsqueeze(0) # get a 1D array
        state = state.to(self.device)
        eps = self.episilon
        if not self.training:
            eps = 1.
        if np.random.randn() <= eps:# greedy policy for training
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].cpu().numpy()
            # action = action[0]
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0, self.num_actions, (n))
            # action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        #update the parameters
        if self.training:
            if self.learn_step_counter % self.update_freq == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter+=1

            #sample batch from memory
            sample_index = np.random.choice(self.memory_capacity, self.batch_size)
            batch_memory = self.memory[sample_index, :]
            batch_state = torch.FloatTensor(batch_memory[:, :self.num_states]).to(self.device)
            batch_action = torch.LongTensor(batch_memory[:, self.num_states:self.num_states+1].astype(int)).to(self.device)
            batch_reward = torch.FloatTensor(batch_memory[:, self.num_states+1:self.num_states+2]).to(self.device)
            batch_next_state = torch.FloatTensor(batch_memory[:, -self.num_states:]).to(self.device)

            #q_eval
            q_eval = self.eval_net(batch_state).gather(1, batch_action)
            q_next = self.target_net(batch_next_state).detach()
            q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss
        return None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def run_batch(self, x, env, labels):
        batch_size = x.shape[0]
        x = x.cpu().detach().numpy().reshape(batch_size, -1)

        total_reward = 0
        # TODO 固定初状态返回root节点，是否可以随机返回一个node？
        batch_state, _ = env.reset(batch_size, labels)
        done = np.zeros(batch_size, dtype=int)
        out = [['fall11'] for _ in range(batch_size)]
        batch_reward = [0. for _ in range(batch_size)]
        dqn_loss = 0
        cnt = 0

        while np.min(done) == 0:
            temp = np.zeros((batch_size, self.num_states))
            for i in range(batch_size):
                if batch_state[i].is_leaf():
                    fc_weight = np.zeros(x[i].shape)
                else:
                    fc_weight = batch_state[i].classifier.state_dict()['1.weight'].cpu().numpy()
                    fc_weight = fc_weight.mean(0)
                temp[i] = np.concatenate((x[i], fc_weight), axis=None)
            batch_action = self.select_action(torch.as_tensor(temp, dtype=torch.float32))
            batch_next_state, reward, done, _ = env.step(batch_size, batch_action)

            for i in range(batch_size):
                if done[i] == 0:
                    if batch_next_state[i].is_leaf():
                        fc_weight = np.zeros(x[i].shape)
                    else:
                        fc_weight = batch_next_state[i].classifier.state_dict()['1.weight'].cpu().numpy()
                        fc_weight = fc_weight.mean(0)
                    next_temp = np.concatenate((x[i], fc_weight), axis=None)
                    self.store_transition(temp[i], batch_action[i], reward[i], next_temp)
                    batch_reward[i] += reward[i]
                    out[i].append(batch_next_state[i].wnid)

                    if self.training and self.memory_counter >= self.memory_capacity:
                        dqn_loss += self.learn()
                        cnt += 1

            batch_state = batch_next_state
            
        total_reward += sum(batch_reward)
        dqn_loss = dqn_loss / cnt if cnt > 0 else dqn_loss
        return out, total_reward, dqn_loss
