import torch

a = torch.tensor([1,4,6])
print(a.shape)
a = a.reshape(1,3)
print(f'Matrix A shape: {a.shape}')

b = torch.tensor([[2,3],[5,8],[7,9]])
print(f'Matrix B shape: {b.shape}')

c =a @ b
print(c.shape)
print(c)