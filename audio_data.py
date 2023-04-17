import torch
import torchaudio
import pytorch_lightning as pl

from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC

from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H

def wave_to_mfcc_transform(sample_rate=8000):
    n_fft = 512
    win_length = None
    n_mels = 128
    n_mfcc = 40

    return MFCC(
    sample_rate=sample_rate,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'mel_scale': 'htk',
    })

def get_wave2vec_model():
    return WAV2VEC2_ASR_LARGE_LV60K_960H.get_model()

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size: int = 64, resample_freq = None, wave_transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resample_freq = resample_freq
        self.padding_freq = resample_freq if resample_freq is not None else 16000
        self.wave_transform = wave_transform
        labels_to_predict = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
        labels_to_predict = dict(zip(labels_to_predict, range(len(labels_to_predict))))
        self.labels_to_predict = labels_to_predict
        self.unknown_label = len(labels_to_predict)


    def get_collate_fn(self):
        def collate_fn(batch):
            inputs = []
            targets = []
            with torch.no_grad():
                for elem in batch:
                    # waveform, sample_rate, label, speaker_id, utterance_number = elem
                    waveform, _, label, _, _ = elem
                    
                    if self.resample_freq is not None:
                        waveform = torchaudio.transforms.Resample(16000, self.resample_freq)(waveform)

                    #if waveform.shape[1] > 8000:
                    #    raise ValueError("Waveform is too long")

                    bytes_to_pad = self.padding_freq - waveform.shape[1]
                    
                    if bytes_to_pad > 0:
                        waveform = torch.nn.functional.pad(waveform, (0, bytes_to_pad), mode='constant', value=0)

                    if self.wave_transform is not None:
                        waveform = self.wave_transform(waveform)

                    if label in self.labels_to_predict:
                        label = self.labels_to_predict[label]
                    else:
                        label = self.unknown_label
                    
                    inputs.append(waveform)
                    targets.append(label)
            
            inputs = torch.stack(inputs)
            targets = torch.tensor(targets)

            return inputs, targets
            
        return collate_fn

    def prepare_data(self) -> None:
        super().prepare_data()
        version = "speech_commands_v0.01"
        self.speechcommands_train = SPEECHCOMMANDS(self.data_dir, url=version, download=True, subset="training")
        self.speechcommands_test = SPEECHCOMMANDS(self.data_dir, url=version, download=True, subset="testing")
        self.speechcommands_val = SPEECHCOMMANDS(self.data_dir, url=version, download=True, subset="validation")


    def setup(self, stage=None):
        return None
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_train, batch_size=self.batch_size, collate_fn=self.get_collate_fn())

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_val, batch_size=self.batch_size, collate_fn=self.get_collate_fn())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_test, batch_size=self.batch_size, collate_fn=self.get_collate_fn())
    
    def data_dimensions(self):
        return None



model = get_wave2vec_model()
fun = lambda waveform: model.extract_features(waveform)[0]
audio_module = AudioDataModule(data_dir='./', batch_size=64)
audio_module.prepare_data()
audio_module.setup()

for elem in audio_module.val_dataloader():
    with torch.no_grad():
        print(fun(elem[0][0])[0].shape)
    break