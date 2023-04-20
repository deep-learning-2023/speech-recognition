import numpy as np
import webrtcvad

def remove_silence(audio_data, sample_rate):
    """Removes silence from the beginning and end of an audio file using the WebRTC VAD module."""
    
    vad = webrtcvad.Vad()
    vad.set_mode(2)

    frame_size = sample_rate // 100
    frames = frame_generator(frame_size, audio_data)
    is_speech_list = [vad.is_speech(frame.tobytes(), sample_rate) for frame in frames]
    if True in is_speech_list:
        first_true = is_speech_list.index(True)
        last_true = len(is_speech_list) - is_speech_list[::-1].index(True) - 1
        return np.concatenate(frames[first_true:last_true + 1])
    return np.array([]) # silence

def frame_generator(frame_size, audio_data):
    num_frames = len(audio_data) // frame_size
    return [audio_data[i * frame_size:(i + 1) * frame_size] for i in range(num_frames)]
