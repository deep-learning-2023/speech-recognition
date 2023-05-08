import torch
import torchaudio
import pytorch_lightning as pl

from myspeechcommands import MYSPEECHCOMMANDS
from torchaudio.transforms import MFCC

from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_LV60K_960H
import cv2
import numpy as np


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 64,
        data_transform=None,
        label_subset: list[str] = None,
        collate_fn=torch.utils.data.default_collate,
        wav2vec: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transform = data_transform
        self.label_subset = label_subset
        self.wav2vec = wav2vec
        self.collate_fn = collate_fn

    def cloned_subset(self):
        if self.label_subset is None:
            return None
        else:
            return self.label_subset.copy()

    def prepare_data(self) -> None:
        super().prepare_data()
        self.speechcommands_train = MYSPEECHCOMMANDS(
            self.data_dir,
            subset="training",
            transform=self.data_transform,
            labels_subset=self.cloned_subset(),
            wav2vec_transformed=self.wav2vec,
        )
        self.speechcommands_test = MYSPEECHCOMMANDS(
            self.data_dir,
            subset="testing",
            transform=self.data_transform,
            labels_subset=self.cloned_subset(),
            wav2vec_transformed=self.wav2vec,
        )
        self.speechcommands_val = MYSPEECHCOMMANDS(
            self.data_dir,
            subset="validation",
            transform=self.data_transform,
            labels_subset=self.cloned_subset(),
            wav2vec_transformed=self.wav2vec,
        )

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.speechcommands_train,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.speechcommands_val,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.speechcommands_test,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def get_data_dimensions(self):
        return self.speechcommands_train.__getitem__(1)[0].shape

    def get_label_name(self, n: int):
        return self.speechcommands_train.get_label(n)


def pad_to(x, length):
    if x.shape[1] > length:
        raise ValueError("Waveform is too long")

    bytes_to_pad = length - x.shape[1]

    if bytes_to_pad > 0:
        x = torch.nn.functional.pad(x, (0, bytes_to_pad), mode="constant", value=0)

    return x


def MFCC_transform():
    n_fft = 512
    n_mels = 256
    n_mfcc = 40

    mfcc = MFCC(
        sample_rate=8000,
        # n_mfcc=n_mfcc,
        # melkwargs={
        #     "n_fft": n_fft,
        #     "n_mels": n_mels,
        #     "mel_scale": "htk",
        # },
    )

    def tf(waveform):
        waveform = torchaudio.transforms.Resample(16000, 8000)(waveform)

        waveform = pad_to(waveform, 8000)

        ficzury = mfcc(waveform)

        ficzury = torch.squeeze(ficzury, dim=0)

        ficzury = torch.swapaxes(ficzury, 0, 1)

        return ficzury

    return tf


def wave2vec_transform():
    model = WAV2VEC2_ASR_LARGE_LV60K_960H.get_model()

    def tf(waveform):
        waveform = waveform
        waveform = pad_to(waveform, 16000)
        with torch.no_grad():
            ficzury, _ = model(waveform)
        ficzury = torch.squeeze(ficzury, dim=0)
        return ficzury

    return tf


def spectrogram_transform():
    def tf(waveform):
        waveform = torchaudio.transforms.Resample(16000, 8000)(waveform)
        waveform = pad_to(waveform, 8000)
        spectrogram = torchaudio.transforms.Spectrogram(100)(waveform)
        # ficzury = torch.squeeze(ficzury, dim=0)
        image = spectrogram.squeeze().transpose(0, 1).numpy()
        img = (image * 1000).astype("uint16")
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresholded = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        moments = cv2.moments(thresholded)
        huMoments = cv2.HuMoments(moments)
        # log transfrom
        for i in range(0, 7):
            if huMoments[i] != 0:
                huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i]))
        features = torch.tensor(huMoments, dtype=torch.float32)
        print(features.shape)
        return features

    return tf


def discard_if_unknown(x):
    # x is timeseries of probabilities
    # discard entry if probability of the first class is higher than 0.5
    tnsrs = []
    y = torch.argmax(x, dim=1)
    for i in range(x.shape[0]):
        if y[i] != 0:
            tnsrs.append(x[i][1:])
    if len(tnsrs) < 1:
        tnsrs.append(x[0][1:])
    ret = torch.stack(tnsrs)
    return ret


def pad_to_max_length_collator(batch):
    # batch is a list of tuples (x, y)
    # x is a tensor of shape (sequence_length, input_size)
    # y is a tensor of shape (1)
    # pad all sequences to the length of the longest sequence
    # and stack them into a tensor of shape (batch_size, sequence_length, input_size)
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.as_tensor(y)
    return x, y
