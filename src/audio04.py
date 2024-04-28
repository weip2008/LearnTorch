import torch
import torchaudio
import librosa

yesno_data = torchaudio.datasets.YESNO('data/', download=False)
data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=False)

for data in data_loader: # data_loader is an iterable
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break                                          

import matplotlib.pyplot as plt

print(data[0][0].numpy())
waveform = data[0].reshape(1,-1)
plt.figure()
plt.plot(waveform.t().numpy())
plt.show()