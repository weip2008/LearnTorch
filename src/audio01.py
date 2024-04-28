"""
pip install torchaudio
"""

import torch
import torchaudio

print(torch.__version__)
print(f'CUDA is available: {torch.cuda.is_available()}')
'''
Torch CUDA is a package in the PyTorch library that allows you to use GPUs 
to accelerate computations in deep learning models.
'''
x = torch.rand(5, 3)
print(x)

torchaudio.datasets.YESNO(
     root='./data',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)