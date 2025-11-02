import torch
import torchaudio
import IPython
import scipy
import record
import process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file="sum/sum1.wav"


record.record_audio("transcribe")
IPython.display.Audio(file)
file="transcrib/transcribe.wav"
fs, audio_samples = scipy.io.wavfile.read(file)
waveform, sample_rate = torchaudio.load(file)
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print(fs)
print(waveform)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    print(waveform)
model = bundle.get_model().to(device)
with torch.inference_mode():
    emission, _ = model(waveform)
print(emission)

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)
t=transcript.split("|")
print(t)
name=t[2]
print(name)


def transcribe(file):
    waveform, sample_rate = torchaudio.load(file)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    waveform = waveform.to(device)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    model = bundle.get_model().to(device)
    with torch.inference_mode():
        emission, _ = model(waveform)
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])
    print(transcript)
    t=transcript.split("|")
    print(t)
    name=t[2]
    print(name)
    return name
    
    
