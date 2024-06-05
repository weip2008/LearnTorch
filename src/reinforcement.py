import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        
        # 计算卷积层输出的特征图的大小
        conv_output_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_shape[1])

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 创建DQN智能体
class DQNAgent:
    def __init__(self, input_shape, output_shape, replay_buffer_capacity=10000, batch_size=64, gamma=0.99, lr=0.0001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(input_shape, output_shape).to(self.device)
        self.target_dqn = DQN(input_shape, output_shape).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.dqn(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(3)  # 假设有3个动作
        return action

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).unsqueeze(1).to(self.device)  # 确保 action 是 int64 类型并加上维度
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.dqn(state).gather(1, action).squeeze(1)
        next_q_values = self.target_dqn(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

# 示例用法
if __name__ == "__main__":
    input_shape = (1, 6, 200)
    output_shape = (500, 3)
    agent = DQNAgent(input_shape, output_shape)

    # 假设你有一些状态、动作、奖励、下一个状态和完成标志的数据
    # 这里只是示例，实际数据应该来自环境
    state = np.random.rand(64, 1, 6, 200)
    action = np.random.randint(0, 3, size=64)
    reward = np.random.rand(64)
    next_state = np.random.rand(64, 1, 6, 200)
    done = np.random.randint(0, 2, size=64)

    # 将数据推入经验回放缓冲区
    for i in range(64):
        agent.replay_buffer.push(state[i], action[i], reward[i], next_state[i], done[i])

    # 训练DQN智能体
    for _ in range(1000):
        agent.train_step()
        if _ % 10 == 0:
            agent.update_target()
