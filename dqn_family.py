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
import sys
import os
from segment_tree import MinSegmentTree, SumSegmentTree



class DQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
    
    
class DuelingDQN(nn.Module):
    def __init__(self, args, input_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.value_stream = nn.Sequential(
            nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.mlp_hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(args.mlp_hidden_dim // 2, action_dim)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        v = self.value_stream(x) # state value
        a = self.advantage_stream(x) # advantages
        q = v + (a - a.mean())
        return q


class ReplayBuffer:
    def __init__(self, args, state_dim):
        self.state_dim = state_dim
        self.memory_capacity = args.memory_capacity
        self.batch_size = args.batch_size

        self.ptr = 0
        self.size = 0
        
        # Initialize replay buffer
        self.buffer = {'s': np.zeros([self.memory_capacity, self.state_dim]),
                       'a': np.zeros([self.memory_capacity, 1]),
                       'r': np.zeros([self.memory_capacity, 1]),
                       's_prime': np.zeros([self.memory_capacity, self.state_dim]),
                       'done': np.ones([self.memory_capacity, 1])
                       }

    def storeSample(self, s, a, r, s_prime, done):
        self.buffer['s'][self.ptr] = s
        self.buffer['a'][self.ptr] = a
        self.buffer['r'][self.ptr] = r
        self.buffer['s_prime'][self.ptr] = s_prime
        self.buffer['done'][self.ptr] = done

        # Rewrite the experience from the begining like FIFO style rather than pop
        self.ptr = (self.ptr + 1) % self.memory_capacity
        self.size = min(self.size + 1, self.memory_capacity)

    def sample(self):
        # Uniform batch sampling
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        mini_batch = {}
        for key in self.buffer.keys():
            if key == 'a':
                mini_batch[key] = torch.tensor(self.buffer[key][indices], dtype=torch.int64)
            else:
                mini_batch[key] = torch.tensor(self.buffer[key][indices], dtype=torch.float32)
                
        return mini_batch
    
    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Modified version of https://github.com/keep9oing/DQN-Family/blob/master/DQN_PER.py
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
    """
    
    def __init__(self, args, state_dim, alpha=0.4, beta=0.4):
        """Initialization."""
        assert alpha >= 0
        assert beta >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(args, state_dim)
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha
        self.beta = beta
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.memory_capacity:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def storeSample(self, s, a, r, s_prime, done):
        """Put experience and priority"""
        super().storeSample(s, a, r, s_prime, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.memory_capacity

    def sample(self):
        """Sample a batch of experiences"""
        assert len(self) >= self.batch_size
        
        indices = self._sample_proportional()
        mini_batch = {}
        for key in self.buffer.keys():
            if key == 'a':
                mini_batch[key] = torch.tensor(self.buffer[key][indices], dtype=torch.int64)
            else:
                mini_batch[key] = torch.tensor(self.buffer[key][indices], dtype=torch.float32)
        weights = np.array([self._calculate_weight(i, self.beta) for i in indices])
        mini_batch["weights"] = torch.from_numpy(weights).type(torch.float32)
        mini_batch["indices"] = indices

        return mini_batch

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions"""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            if priority <= 0:
                print(priority)
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        """Sample indices based on proportions"""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # Get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # Calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight


class Trainer:
    def __init__(self, args):
        # Gym environment
        self.env = gym.make(args.env)
        self.state_dim = self.env.observation_space.shape[0]
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
        self.use_per = args.use_per
        if args.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(args, self.state_dim, alpha=args.alpha, beta=args.beta)
        else:
            self.replay_buffer = ReplayBuffer(args, self.state_dim)
        self.enough_memory_size_to_train = args.enough_memory_size_to_train

        # Deep Q-network
        self.network_type = args.network_type
        
        if args.network_type == "dqn" or args.network_type == "ddqn": # Double-DQN
            self.q_network = DQN(args, self.state_dim, self.action_dim)
            self.target_q_network = DQN(args, self.state_dim, self.action_dim)
        elif args.network_type == "dueling-dqn" or args.network_type == "d3qn": 
            self.q_network = DuelingDQN(args, self.state_dim, self.action_dim)
            self.target_q_network = DuelingDQN(args, self.state_dim, self.action_dim)    
        else:
            print(">>> Selected model {} is invalid".format(args.network_type))
            sys.exit()
        
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Target network update
        self.target_update_period = args.target_update_period
        self.use_soft_update = args.use_soft_update
        self.tau = args.tau

        # Optimizer        
        self.lr = args.lr
        # self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.lr, weight_decay=1e-4)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Tensorboard results
        property = "seed_" + str(args.seed) + "_" + args.network_type + "_epi_" + str(args.episodes) \
                    + "_lr_" + str(args.lr) + "_soft_" + str(args.use_soft_update) + "_per_" + str(args.use_per)
        print(">>> Train property: ", property)
        path = os.path.join("runs", args.env, property)
        self.writer = SummaryWriter(log_dir=path)

    # epsilon greedy
    def chooseAction(self, s):
        q_values = self.q_network(s)
        coin = random.random()
        if coin < self.epsilon:
            return random.randint(0,1)
        else : 
            return q_values.argmax().item()

    def train(self):
        mini_batch = self.replay_buffer.sample()
        
        if self.use_per:
            s, a, r, s_prime, done, weights, indices = mini_batch.values()
        else:
            s, a, r, s_prime, done = mini_batch.values()
        
        if self.network_type == "dqn" or self.network_type == "dueling-dqn":
            q = self.q_network(s).gather(1, a)
            with torch.no_grad():
                max_q_prime = self.target_q_network(s_prime).max(1)[0].unsqueeze(1)
                TD_target = r + self.gamma * max_q_prime * (1 - done)
        elif self.network_type == "ddqn" or self.network_type == "d3qn":
            q = self.q_network(s).gather(1, a)
            with torch.no_grad():
                optimal_a_prime = self.q_network(s_prime).max(1)[1].unsqueeze(1)
                target_q_prime = self.target_q_network(s_prime).gather(1, optimal_a_prime)
                TD_target = r + self.gamma * target_q_prime * (1 - done)
        else:
            sys.exit()

        # TD target must be fixed (use .detach() or no_grad())
        if self.use_per:
            elementwise_loss = F.smooth_l1_loss(q, TD_target, reduction="none")
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = F.smooth_l1_loss(q, TD_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        if self.use_per:
            loss_for_prior = elementwise_loss.detach().cpu().numpy()
            new_priorites = loss_for_prior + sys.float_info.epsilon
            self.replay_buffer.update_priorities(indices, new_priorites)

        
        return loss.data.item()

    def getLearningRate(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def updateTarget(self, episode):
        if self.use_soft_update:
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            if episode != 0 and episode % self.target_update_period == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

    def learn(self):
        ##############################################
        # YOU MUST FIX "num_steps" EVALUATION PARAMETERS. If you don't you can get penalty
        num_steps = 1000
        ##############################################

        score = 0.0
        step = 0
        render = False

        for episode in range(self.episodes):
            # Initialize
            s = self.env.reset()

            for _ in range(num_steps):
                # if render:
                #     env.render()
                step += 1
                
                a = self.chooseAction(torch.from_numpy(s).float())      
                s_prime, r, done, _ = self.env.step(a)
                self.replay_buffer.storeSample(s, a, r/100, s_prime, done)
                s = s_prime

                score += r
                if done:
                    break

                if len(self.replay_buffer) > self.enough_memory_size_to_train:
                    loss = self.train()
                    self.writer.add_scalar("loss/train", loss, global_step=step)
                    self.updateTarget(episode)

            # Epsilon decaying
            self.epsilon = max(self.min_epsilon,  self.epsilon * self.epsilon_decay_rate)

            if ((episode+1) % 20) == 0:
                mean_20ep_reward = round(score/20, 1)
                print("train episode: {}, average reward: {:.1f}, buffer size: {}, epsilon: {:.1f}%".format(episode+1, mean_20ep_reward, len(self.replay_buffer), self.epsilon*100))
                self.writer.add_scalar("score/train", mean_20ep_reward, global_step=episode)

                # Initialize score every 20 episodes
                score = 0.0

        self.env.close()
        self.writer.flush()
        self.writer.close()

    def evaluate(self, episode_rewards):
        # You can plot the another evaluation values.
        # But, you should include the "episode_return".
        plt.xlabel("Num Episode")
        plt.ylabel("Episode returns")
        plt.plot(episode_rewards, label='episode_return')
        plt.legend()
        plt.savefig('DQN Episode returns')
        plt.show()
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Gym environment
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment type (CartPole-v1, Acrobot-v1, MountainCar-v0)")
    
    # Deep Q-network
    parser.add_argument("--network_type", type=str, default="dqn", help="Deep Q-network type (dqn, ddqn, dueling-dqn, d3qn)")
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='MLP layer hidden dimension')
    parser.add_argument('--rnn_hidden_dim', type=int, default=128, help='RNN layer hidden dimension')

    parser.add_argument('--episodes', default=2000, type=int, help='Number of training episode (epochs)')
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Training parameters
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')

    # Epsilon-greedy policy
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_rate", type=float, default=0.995, help="Epsilon decaying rate")    

    # Target network update
    parser.add_argument("--target_update_period", type=int, default=20, help="Target network update period")
    parser.add_argument("--use_soft_update", action="store_true", help="Use hard target network update")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft target update parameter")

    # Experience replay
    parser.add_argument('--memory_capacity', default=50000, type=int, help='Replay memory capacity')
    parser.add_argument('--enough_memory_size_to_train', default=2000, type=int, help='Batch size')
    # Prioritized xperience replay (PER)
    parser.add_argument("--use_per", action="store_true", help="Use prioritized experience replay")
    parser.add_argument("--alpha", type=float, default=0.7, help="Prioritized experience replay parameter")    
    parser.add_argument("--beta", type=float, default=0.5, help="Prioritized experience replay parameter")    
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Training
    trainer = Trainer(args)
    trainer.learn()

if __name__ == '__main__':
    main()