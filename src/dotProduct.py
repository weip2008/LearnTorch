import torch

a = torch.tensor([1,4,6])
print(a.shape)
a = a.reshape(1,3)
print(f'{a.shape}', end=' ')

b = torch.tensor([[2,3],[5,8],[7,9]])
print(f'{b.shape}')

c =a @ b
# c =b @ a
print(c.shape) # a=(1,3); b=(3,2); c⟹(1,2)
d = c.squeeze()
print(d.shape)
print(c)