import os
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio.functional as F
from model import Stacked_Conv

AUDIO_PATH = "/home/florent/Dataset/CSTR_VCTK/wav48_silence_trimmed/p228/"
audio, sr = sf.read(os.path.join(AUDIO_PATH, "p228_022_mic2.flac"), dtype="float32")
audio = torch.from_numpy(audio)
audio_16k = F.resample(audio, orig_freq=sr, new_freq=16000)
audio = F.mu_law_encoding(audio_16k, 256)
audio = nn.functional.one_hot(audio, num_classes=256).float()
audio = audio.T.unsqueeze(0)

print(audio.shape)

def test_causality():
    model = Stacked_Conv()

    model.eval()

    x1 = audio.clone()
    x2 = audio.clone()

    t = 1000

    future_len = x2.shape[-1] - (t + 1)
    random_classes = torch.randint(0, 256, (1, future_len))
    random_future = torch.nn.functional.one_hot(random_classes, num_classes=256).float()
    random_future = random_future.permute(0, 2, 1)  # [1, 256, future_len]
    print(random_future)
    x2[:, :, t+1:] = random_future
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)

    diff = (y1[:, :, :t+1] - y2[:, :, :t+1]).abs().max().item()
    print(diff)

test_causality()
