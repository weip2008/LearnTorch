import torch
import torch.nn as nn

# 定义生成器（Generator）
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 定义判别器（Discriminator）
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 定义GAN模型
class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # 生成器生成图像
        generated_images = self.generator(x)

        # 判别器对生成的图像进行判别
        discriminator_output = self.discriminator(generated_images)

        return generated_images, discriminator_output

# 设置超参数
input_size = 100  # 输入噪声向量的大小
hidden_size = 128  # 隐藏层大小
output_size = 784  # 生成的图像大小（28x28）

# 创建生成器和判别器
generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(output_size, hidden_size, 1)

# 创建GAN模型
gan_model = GAN(generator, discriminator)

# 打印GAN模型结构
print(gan_model)
