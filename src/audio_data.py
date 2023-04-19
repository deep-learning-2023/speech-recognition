import torch
import torchaudio
import pytorch_lightning as pl

from myspeechcommands import MYSPEECHCOMMANDS
from torchaudio.transforms import MFCC

from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H

class AudioDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size: int = 64, data_transform=None, label_subset: list[str] = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transform = data_transform
        self.label_subset = label_subset

    def prepare_data(self) -> None:
        super().prepare_data()
        self.speechcommands_train = MYSPEECHCOMMANDS(self.data_dir, subset="training", transform=self.data_transform, labels_subset=self.label_subset)
        self.speechcommands_test = MYSPEECHCOMMANDS(self.data_dir, subset="testing", transform=self.data_transform, labels_subset=self.label_subset)
        self.speechcommands_val = MYSPEECHCOMMANDS(self.data_dir, subset="validation", transform=self.data_transform, labels_subset=self.label_subset)


    def setup(self, stage=None):
        return None
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.speechcommands_test, batch_size=self.batch_size)
    
    def get_data_dimensions(self):
        return self.speechcommands_train.__getitem__(1)[0].shape
    
    def get_label_name(self, n: int):
        return self.speechcommands_train.get_label(n)

def pad_to(x, length):
    if x.shape[1] > length:
        raise ValueError("Waveform is too long")

    bytes_to_pad = length - x.shape[1]

    if bytes_to_pad > 0:
        x = torch.nn.functional.pad(x, (0, bytes_to_pad), mode='constant', value=0)

    return x


def MFCC_transform():
    n_fft = 512
    n_mels = 128
    n_mfcc = 40

    mfcc = MFCC(
    sample_rate=8000,
    n_mfcc=n_mfcc,
    melkwargs={
      'n_fft': n_fft,
      'n_mels': n_mels,
      'mel_scale': 'htk',
    })

    def tf(waveform):
        waveform = torchaudio.transforms.Resample(16000, 8000)(waveform)

        waveform = pad_to(waveform, 8000)

        ficzury = mfcc(waveform)

        ficzury = torch.squeeze(ficzury, dim=0)

        return ficzury

    return tf

def wave2vec_transform():
    model = WAV2VEC2_ASR_LARGE_LV60K_960H.get_model().to('cuda')

    def tf(waveform):
        waveform = waveform.to('cuda')
        waveform = pad_to(waveform, 16000)
        with torch.no_grad():
            ficzury, _ = model(waveform)
        ficzury = torch.squeeze(ficzury, dim=0)
        return ficzury
            

    return tf
