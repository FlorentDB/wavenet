import os
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio.functional as F


AUDIO_PATH = "/home/florent/Dataset/CSTR_VCTK/wav48_silence_trimmed/p228/"
audio, sr = sf.read(os.path.join(AUDIO_PATH, "p228_022_mic2.flac"), dtype="float32")
audio = torch.from_numpy(audio)
audio_16k = F.resample(audio, orig_freq=sr, new_freq=16000)
audio = F.mu_law_encoding(audio_16k, 256)
audio = nn.functional.one_hot(audio, num_classes=256)
audio = audio.T.unsqueeze(0)
print(audio.shape)

