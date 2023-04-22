from torchaudio.datasets import SPEECHCOMMANDS
import torch
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_data import pad_to

dataset = SPEECHCOMMANDS("./", url="speech_commands_v0.01", download=True, subset=None)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
model = WAV2VEC2_ASR_LARGE_LV60K_960H.get_model().to('cuda')
for data in tqdm(loader):
    waveform, fname, sample_rate, label, speaker_id, utterance_number = data
    if label == '_background_noise_':
        continue

    file = './SpeechCommands/' + fname[0][:-3] + 'pt'
    waveform = waveform.to('cuda')
    waveform = torch.squeeze(waveform, dim=1)
    waveform = pad_to(waveform, 16000)
    with torch.no_grad():
        ficzury, _ = model(waveform)
    ficzury = torch.squeeze(ficzury, dim=0)
    ficzury = ficzury.to('cpu')
    torch.save(ficzury, file)