import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        global window,columns,batch_global
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(columns*window, window),
            nn.ReLU(),  # Rectified Linear Unit
            nn.Linear(window, columns),
            nn.ReLU(),
            nn.Linear(columns, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class CNN(nn.Module):
    """
    卷积神经网络
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 300, 120)  # Assuming input size of 1200, after two maxpooling layers
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)  # 3 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 300)  # Flatten the output of the convolutional layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RNN(nn.Module):
    """
    Recurrent Neural Network
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(2)
        x = x.view(batch_size, seq_length, -1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        attention_input = torch.cat((encoder_outputs, h), dim=2)
        attention_energies = self.score(attention_input)
        return nn.functional.softmax(attention_energies, dim=1)
    
    def score(self, attention_input):
        energy = torch.tanh(self.attention(attention_input))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(attention_input.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        encoder_outputs, (hidden, cell) = self.encoder(x)
        hidden = hidden[-1]
        attention_weights = self.attention(hidden, encoder_outputs)
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs).squeeze(1)
        rnn_output, (hidden, cell) = self.decoder(context.unsqueeze(1))
        output = self.fc(rnn_output.squeeze(1))
        return output

class Attention(nn.Module):
    """
    Attension Machanics
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply linear layer to input
        x = self.fc(x)
        batch_size, seq_len, _ = x.size()

        # Calculate attention scores
        attn_energies = self.attn(x)
        attn_energies = self.tanh(attn_energies)
        attn_energies = attn_energies.matmul(self.v)

        # Apply softmax to get attention weights
        attn_weights = self.softmax(attn_energies)

        # Apply attention weights to input
        context = torch.einsum("ijk,ij->ik", x, attn_weights)

        # Output prediction
        output = self.out(context)
        return output

class TransformModel(nn.Module):
    """
    Transform Model
    """
    def __init__(self):
        super(TransformModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 200, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AutoEncoder(nn.Module):
    """
    Auto Encoders
    """
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.encoder_conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.encoder_fc = nn.Linear(3200, 256)  # Change the number of output features to 2400
        
        # Decoder
        self.decoder_fc = nn.Linear(256, 3200)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(16, 3, 3, stride=2, output_padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x_shape = x.size()  # Get the shape of the tensor
        x = x.view(x_shape[0], -1)  # Flatten the tensor to (batch_size, num_channels * height * width)
        x = F.relu(self.encoder_fc(x))
        
        # Decoder
        x = F.relu(self.decoder_fc(x))
        x = x.view(x_shape)  # Reshape back to original shape
        x = F.relu(self.decoder_conv2(x))
        x = torch.sigmoid(self.decoder_conv1(x))

        return x

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, z):
        x_fake = self.generator(z)
        y_fake = self.discriminator(x_fake)
        return x_fake, y_fake

# 加载MNIST数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
  
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
