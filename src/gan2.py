import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
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

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 初始化生成器和判别器
generator = Generator(input_size=100, hidden_size=128, output_size=784)
discriminator = Discriminator(input_size=784, hidden_size=128, output_size=1)

# 初始化GAN模型
gan_model = GAN(generator, discriminator)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# 训练GAN模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        # 准备数据
        real_images, _ = data
        real_images = real_images.view(-1, 28 * 28)
        batch_size = real_images.size(0)
        real_label = torch.full((batch_size, 1), 1.0)
        fake_label = torch.full((batch_size, 1), 0.0)
        noise = torch.randn(batch_size, 100)

        # 训练判别器
        optimizer_d.zero_grad()
        output_real = discriminator(real_images)
        loss_real = criterion(output_real, real_label)
        loss_real.backward()

        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        loss_fake = criterion(output_fake, fake_label)
        loss_fake.backward()
        optimizer_d.step()

        # 训练生成器
        optimizer_g.zero_grad()
        output = discriminator(fake_images)
        loss_g = criterion(output, real_label)
        loss_g.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Batch Step [{i}/{len(trainloader)}], "
                  f"Loss D: {loss_real + loss_fake:.4f}, Loss G: {loss_g:.4f}")

# 生成一些图像进行展示
with torch.no_grad():
    noise = torch.randn(16, 100)
    fake_images = generator(noise).view(-1, 28, 28)

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(fake_images[i].cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.show()
