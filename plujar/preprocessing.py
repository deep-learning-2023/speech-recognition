import os
from scipy.io import wavfile
from scipy import signal
import numpy as np
import librosa

from remove_silence import remove_silence

new_sample_rate = 8000

def _process_background_noise_dir(train_audio_path):
  wavs = [f for f in os.listdir(os.path.join(train_audio_path, '_background_noise_')) if f.endswith('.wav')]
  xs = []
  for wav in wavs:
      sample_rate, samples = wavfile.read(os.path.join(train_audio_path, '_background_noise_', wav))
      # resampled = np.int16(np.rint(signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))))
      resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
      for i in range(0, resampled.shape[0], new_sample_rate):
          new_samples = resampled[i:i+new_sample_rate]
          if new_samples.shape[0] < new_sample_rate:
              new_samples = np.pad(new_samples, (0, new_sample_rate - new_samples.shape[0]), "constant")
          xs.append(new_samples)
  return xs

def get_raw_data(train_audio_path = './data/train/audio/', limit = 10):
  dirs = [f for f in os.listdir(train_audio_path) if os.path.isdir(os.path.join(train_audio_path, f))]
  dirs = [d for d in dirs if not d.startswith('_')]
  dirs.sort()
  xs = []
  ys = []
  labels = ['others']
  for index, dir in enumerate(dirs):
      wavs = [f for f in os.listdir(os.path.join(train_audio_path, dir)) if f.endswith('.wav')]
      counter = 0
      labels.append(str(dir))
      for wav in wavs:
          counter += 1
          if counter > limit: continue
          sample_rate, samples = wavfile.read(os.path.join(train_audio_path, dir, wav))
          samples = remove_silence(samples, sample_rate)
          if samples.shape[0] == 0:
              continue # skip silence
          # below is the line to be used when wav serialization is needed
          # resampled = np.int16(np.rint(signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))))
          resampled = signal.resample(samples, int(new_sample_rate/sample_rate * samples.shape[0]))
          if resampled.shape[0] < new_sample_rate:
              resampled = np.pad(resampled, (0, new_sample_rate - resampled.shape[0]), "constant")
          xs.append(resampled)
          ys.append(index + 1)

  background_noise = _process_background_noise_dir(train_audio_path)
  for samples in background_noise:
      xs.append(samples)
      ys.append(0)
  return np.vstack(xs), np.array(ys).reshape(-1, 1), labels

def get_raw_audio_data(xs, ys):
  return xs.reshape((*xs.shape ,1)), ys

def get_raw_spectrogram(xs, ys):
  def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
  xs = [log_specgram(x, new_sample_rate)[2] for x in xs]
  return np.concatenate(xs).reshape((len(xs), *xs[0].shape)), ys


def get_mel_spectrogram(xs, ys):
  new_xs = []
  for x in xs:
    S = librosa.feature.melspectrogram(y=x, sr=new_sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    new_xs.append(log_S)

  return np.concatenate(new_xs).reshape((len(new_xs), *new_xs[0].shape)), ys

def get_mfcc(xs, ys):
  new_xs = []
  for x in xs:
    S = librosa.feature.melspectrogram(y=x, sr=new_sample_rate, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, sr=new_sample_rate, n_mfcc=13)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    new_xs.append(delta2_mfcc)

  return np.concatenate(new_xs).reshape((len(new_xs), *new_xs[0].shape)), ys