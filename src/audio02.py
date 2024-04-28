"""
pip install librosa
change the sound frequency from 16000 to 8000 by resample function
"""
import torch
import torchaudio
import librosa

# Load audio file with librosa
audio_file_path = 'data/waves_yesno/0_0_0_0_1_1_1_1.wav'
audio_data, sample_rate = librosa.load(audio_file_path, sr=16000)

# Convert audio data to a torch tensor
audio_tensor = torch.tensor(audio_data)
print(audio_tensor.shape)
# Resample the tensor to match the sample rate expected by torchaudio
resampler = torchaudio.transforms.Resample(sample_rate, 8000)
resampled_tensor = resampler(audio_tensor)

print(resampled_tensor)
# Print information about the resampled tensor
print('Resampled tensor shape:', resampled_tensor.shape)
print('Resampled tensor sample rate:', resampler.new_freq)

"""
Sample rate, also known as sampling frequency, 
refers to the number of samples of audio that are captured per second. 
"""