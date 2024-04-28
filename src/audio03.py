import torch
import torchaudio
import librosa

# * ``download``: If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
#
# Let’s access our Yesno data:
#

# A data point in Yesno is a tuple (waveform, sample_rate, labels) where labels
# is a list of integers with 1 for yes and 0 for no.
yesno_data = torchaudio.datasets.YESNO('data/', download=False)

# Pick data point number 3 to see an example of the the yesno_data:
n = 3
waveform, sample_rate, labels = yesno_data[n]
print("\nWaveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))
print(f'tensor shape: {waveform.shape}, total time {waveform.shape[1]/sample_rate} in seconds.')
print(waveform.shape)