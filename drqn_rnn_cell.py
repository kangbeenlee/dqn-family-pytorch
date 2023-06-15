import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import collections
import random
import gym
import argparse
import time
import sys
import os



class DRQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.RNNCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, action_dim)

    def forward(self, x, h):
        x = F.relu(self.fc1(x))
        h = self.rnn(x, h)
        q = self.fc2(h)
        return q, h

class EpisodeReplayBuffer:
    def __init__(self, args, state_dim):
        self.state_dim = state_dim
        self.episode_limit = args.episode_limit
        self.memory_capacity = args.memory_capacity
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.current_size = 0
        self.buffer = {'s': np.zeros([self.memory_capacity, self.episode_limit + 1, self.state_dim]),
                       'a': np.zeros([self.memory_capacity, self.episode_limit, 1]),
                       'r': np.zeros([self.memory_capacity, self.episode_limit, 1]),
                       'done': np.ones([self.memory_capacity, self.episode_limit, 1]),  # Note: We use 'np.ones' to initialize 'done'
                       'active': np.zeros([self.memory_capacity, self.episode_limit, 1])
                       }
        self.episode_len = np.zeros(self.memory_capacity)

    def storeTransition(self, epi_step, s, a, r, done):
        self.buffer['s'][self.episode_num][epi_step] = s
        self.buffer['a'][self.episode_num][epi_step] = a
        self.buffer['r'][self.episode_num][epi_step] = r
        self.buffer['done'][self.episode_num][epi_step] = done
        
        self.buffer['active'][self.episode_num][epi_step] = 1.0

    def storeLastStep(self, epi_terminal_step, s):
        self.buffer['s'][self.episode_num][epi_terminal_step] = s
        # Record the length of this episode
        self.episode_len[self.episode_num] = epi_terminal_step
        self.episode_num = (self.episode_num + 1) % self.memory_capacity
        self.current_size = min(self.current_size + 1, self.memory_capacity)

    def sample(self):
        # Randomly sampling
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        max_episode_len = int(np.max(self.episode_len[index]))
        batch = {}
        for key in self.buffer.keys():
            if key == 's':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len + 1], dtype=torch.float32)
            elif key == 'a':
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.int64)
            else:
                batch[key] = torch.tensor(self.buffer[key][index, :max_episode_len], dtype=torch.float32)

        return batch, max_episode_len

    def __len__(self):
        return self.current_size

class Trainer:
    def __init__(self, args):
        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("------ Use {} ------".format(self.device))
        
        # Gym environment
        self.env = gym.make(args.env)
        self.state_dim = self.env.observation_space.shape[0] - 2
        self.action_dim = self.env.action_space.n

        # Epsilon-greedy policy parameters
        self.epsilon = args.epsilon
        self.min_epsilon = args.min_epsilon
        self.epsilon_decay_rate = args.epsilon_decay_rate

        # Discount factor
        self.gamma = args.gamma

        # Training paramters
        self.episodes = args.episodes
        self.batch_size = args.batch_size

        # Replay buffer
        self.episode_replay_buffer = EpisodeReplayBuffer(args, self.state_dim)
        self.episode_limit = args.episode_limit
        self.enough_memory_size_to_train = args.enough_memory_size_to_train

        self.network_type = "drqn"
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.q_network = DRQN(args, self.state_dim, self.action_dim).to(self.device)
        self.target_q_network = DRQN(args, self.state_dim, self.action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Target network update
        self.train_update = 0
        self.target_update_period = args.target_update_period
        self.use_soft_update = args.use_soft_update
        self.tau = args.tau

        # Optimizer        
        self.lr = args.lr
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.lr, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Tensorboard results
        property = "RNNCell"
        print(">>> Train property: ", property)
        path = os.path.join("runs", "POMDP", property)
        self.writer = SummaryWriter(log_dir=path)

    # epsilon greedy
    def chooseAction(self, s, h):
        q_values, h = self.q_network(s, h)
        if random.random() < self.epsilon:
            return random.randint(0,1), h
        else : 
            return q_values.argmax().item(), h

    def train(self):
        self.train_update += 1
        mini_batch, episode_len = self.episode_replay_buffer.sample()
        state = mini_batch['s'].to(self.device)
        action = mini_batch['a'].to(self.device)
        reward = mini_batch['r'].to(self.device)
        done = mini_batch['done'].to(self.device)
        active = mini_batch['active'].to(self.device)
        
        inputs = self.getInputs(mini_batch, episode_len).to(self.device) # inputs.shape=(batch_size,episode_len+1,state_dim)

        # Initialize hidden & cell state
        h_state = torch.zeros([self.batch_size, self.rnn_hidden_dim]).to(self.device)
        target_h_state = torch.zeros([self.batch_size, self.rnn_hidden_dim]).to(self.device)
        
        q_evals, q_targets = [], []
        for t in range(episode_len): # t=0,1,2,...(episode_len-1)
            # print(inputs[:, t].shape) # inputs[:, t].shape=(batch_size,state_dim)
            q_eval, h_state = self.q_network(inputs[:, t].to(self.device), h_state) # q_eval.shape=(batch_size,action_dim)
            q_target, target_h_state = self.target_q_network(inputs[:, t + 1].to(self.device), target_h_state)
            q_evals.append(q_eval)  # q_eval.shape=(batch_size, action_dim)
            q_targets.append(q_target)

        # Stack them according to the time (dim=1)
        q_evals = torch.stack(q_evals, dim=1).to(self.device) # q_evals.shape=(batch_size,episode_len,action_dim)
        q_targets = torch.stack(q_targets, dim=1).to(self.device) # q_targets.shape=(batch_size,episode_len,action_dim)
        
        # mini_batch['a'].shape(batch_size,episode_len,1)
        q_a = torch.gather(q_evals, dim=-1, index=action)  # q_evals.shape=(batch_size,episode_len,1)
        
        with torch.no_grad():
            q_target = q_targets.max(dim=-1)[0].unsqueeze(-1).to(self.device)  # q_targets.shape=(batch_size,episode_len)

        # mini_batch['done'].shape=(batch_size,episode_len)
        td_target = reward + self.gamma * q_target * (1 - done)
        td_error = (q_a - td_target.detach())
        
        mask_td_error = td_error * active
        loss = (mask_td_error ** 2).sum() / active.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target update
        if self.use_soft_update:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            if self.train_update % self.target_update_period == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                
        return loss.data.item()

    def getInputs(self, mini_batch, episode_len):
        inputs = []
        inputs.append(mini_batch['s'])
        # if self.add_last_action:
        #     inputs.append(mini_batch['last_onehot_a_n'])
        # if self.add_agent_id:
        #     agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size,episode_len + 1,1,1)
        #     inputs.append(agent_id_one_hot)

        # inputs.shape=(bach_size,episode_len+1,N,input_dim)
        inputs = torch.cat([x for x in inputs], dim=-1)
        
        return inputs

    def learn(self):
        ##############################################
        # YOU MUST FIX "num_steps" EVALUATION PARAMETERS. If you don't you can get penalty
        num_steps = 1000
        ##############################################

        score = 0.0
        step = 0
        render = False

        for epi_num in range(self.episodes):
            # Initialize
            s = self.env.reset()
            obs = s[::2] # Partially observable
            # obs = s

            h_state = torch.zeros([1, self.rnn_hidden_dim]).to(self.device)

            for epi_step in range(self.episode_limit):
                # if render:
                #     env.render()
                step += 1
                
                a, h_state = self.chooseAction(torch.from_numpy(obs).float().unsqueeze(0).to(self.device), h_state)
                s_prime, r, done, _ = self.env.step(a)
                obs_prime = s_prime[::2]
                # obs_prime = s_prime

                # storeTransition(self, epi_step, s, a, r, done):
                self.episode_replay_buffer.storeTransition(epi_step, obs, a, r/100, done)
                
                obs = obs_prime
                score += r
                
                if done:
                    # Store last step
                    self.episode_replay_buffer.storeLastStep(epi_step + 1, obs_prime)
                    break

                if len(self.episode_replay_buffer) > self.enough_memory_size_to_train:
                    loss = self.train()
                    self.writer.add_scalar("loss", loss, global_step=step)

            # Epsilon decaying
            self.epsilon = max(self.min_epsilon,  self.epsilon * self.epsilon_decay_rate)

            if ((epi_num+1) % 20) == 0:
                mean_20ep_reward = round(score/20, 1)
                print("train episode: {}, average reward: {:.1f}, buffer size: {}, epsilon: {:.1f}%".format(epi_num+1, mean_20ep_reward, len(self.episode_replay_buffer), self.epsilon*100))
                self.writer.add_scalar("score", mean_20ep_reward, global_step=epi_num)

                # Initialize score every 20 episodes
                score = 0.0

        self.env.close()
        self.writer.flush()
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Gym environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment type (CartPole-v1, Acrobot-v1, MountainCar-v0)")
    
    # Deep Recurrent Q-network
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='RNN layer hidden dimension')
    parser.add_argument('--episodes', default=2000, type=int, help='Number of training episode (epochs)')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Training parameters
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')

    # Epsilon-greedy policy
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.995, help="Epsilon decaying rate")    

    # Target network update
    parser.add_argument("--target_update_period", type=int, default=20, help="Target network update period")
    parser.add_argument("--use_soft_update", action="store_true", help="Use hard target network update")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")
    
    # Experience replay
    parser.add_argument('--memory_capacity', default=1000, type=int, help='Replay memory capacity')
    parser.add_argument('--episode_limit', default=500, type=int,
                        help='Maximum number of steps per episode (500 for CartPole-v1, Acrobot-v1 and 200 for MountainCar-v0)')
    parser.add_argument('--enough_memory_size_to_train', default=20, type=int, help='Batch size')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Training
    trainer = Trainer(args)
    trainer.learn()

if __name__ == '__main__':
    main()