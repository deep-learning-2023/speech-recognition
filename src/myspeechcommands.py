import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar, _load_waveform

import random

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.01"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
SAMPLE_RATE = 16000
_CHECKSUMS = {
    "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz": "743935421bb51cccdb6bdd152e04c5c70274e935c82119ad7faeec31780d811d",  # noqa: E501
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz": "af14739ee7dc311471de98f5f9d2c9191b18aedfe957f4a6ff791c709868ff58",  # noqa: E501
}

without_noise = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "unknown",
]
unknown_labels = {
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "four",
    "happy",
    "house",
    "marvin",
    "nine",
    "one",
    "seven",
    "sheila",
    "six",
    "three",
    "tree",
    "two",
    "wow",
    "zero",
}
labels_to_predict_mapping = {
    "yes": 0,
    "no": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "on": 6,
    "off": 7,
    "stop": 8,
    "go": 9,
    "_background_noise_": 10,
}
total_good_labels = 11


def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [
                os.path.normpath(os.path.join(root, line.strip())) for line in fileobj
            ]
    return output


def _get_speechcommands_metadata(
    filepath: str, path: str
) -> Tuple[str, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    reldir, filename = os.path.split(relpath)
    _, label = os.path.split(reldir)
    # Besides the officially supported split method for datasets defined by "validation_list.txt"
    # and "testing_list.txt" over "speech_commands_v0.0x.tar.gz" archives, an alternative split
    # method referred to in paragraph 2-3 of Section 7.1, references 13 and 14 of the original
    # paper, and the checksums file from the tensorflow_datasets package [1] is also supported.
    # Some filenames in those "speech_commands_test_set_v0.0x.tar.gz" archives have the form
    # "xxx.wav.wav", so file extensions twice needs to be stripped twice.
    # [1] https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/speech_commands.txt
    speaker, _ = os.path.splitext(filename)
    speaker, _ = os.path.splitext(speaker)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    return relpath, SAMPLE_RATE, label, speaker_id, utterance_number


class MYSPEECHCOMMANDS(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = True,
        subset: Optional[str] = None,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        labels_subset: Optional[list[str]] = None,
        sample_equally: bool = True,
        extension: str = ".wav",
    ) -> None:
        url = URL
        self.transform = transform

        if subset is not None and subset not in ["training", "validation", "testing"]:
            raise ValueError(
                "When `subset` is not None, it must be one of ['training', 'validation', 'testing']."
            )

        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "http://download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)
        self._archive = os.path.join(root, folder_in_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url_to_file(url, archive, hash_prefix=checksum)
                _extract_tar(archive, self._path)
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        if labels_subset is not None:
            # subset_labels_contains_unknown = any([label == 'unknown' for label in labels_subset])
            # labels_subset = labels_subset if not subset_labels_contains_unknown else labels_subset.append(unknown_labels)
            # labels_subset = list(filter(lambda x: x != 'unknown', labels_subset))
            if "unknown" in labels_subset:
                labels_subset.remove("unknown")
                labels_subset += list(unknown_labels)

            for label in labels_subset:
                if (
                    label not in unknown_labels
                    and label not in labels_to_predict_mapping.keys()
                ):
                    raise ValueError(f"{label} is not a valid label.")

            to_predict_labels = list(
                filter(lambda x: x in labels_to_predict_mapping.keys(), labels_subset)
            )
            l_unknown_labels = list(
                filter(lambda x: x in unknown_labels, labels_subset)
            )
            self.local_label_mapping = {
                label: i for i, label in enumerate(to_predict_labels)
            }
            self.local_unknown_idx = len(self.local_label_mapping)
            self.int_to_label = {v: k for k, v in self.local_label_mapping.items()}
            self.int_to_label[self.local_unknown_idx] = "unknown"
            for ul in l_unknown_labels:
                self.local_label_mapping[ul] = self.local_unknown_idx
        else:
            self.local_label_mapping = labels_to_predict_mapping.copy()
            self.int_to_label = {v: k for k, v in self.local_label_mapping.items()}
            self.int_to_label[total_good_labels] = "unknown"
            self.local_unknown_idx = len(self.local_label_mapping)
            for ul in unknown_labels:
                self.local_label_mapping[ul] = self.local_unknown_idx

        if subset == "validation":
            self._walker = [
                w
                for w in _load_list(self._path, "validation_list.txt")
                if any(
                    [
                        label == w.split("/")[2]
                        for label in self.local_label_mapping.keys()
                    ]
                )
            ]
        elif subset == "testing":
            self._walker = [
                w
                for w in _load_list(self._path, "testing_list.txt")
                if any(
                    [
                        label == w.split("/")[2]
                        for label in self.local_label_mapping.keys()
                    ]
                )
            ]
        elif subset == "training":
            excludes = set(
                _load_list(self._path, "validation_list.txt", "testing_list.txt")
            )
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*{extension}"))
            self._walker = [
                w
                for w in walker
                if (
                    HASH_DIVIDER in w
                    and EXCEPT_FOLDER not in w
                    and os.path.normpath(w) not in excludes
                )
                and any(
                    [
                        label == w.split("/")[2]
                        for label in self.local_label_mapping.keys()
                    ]
                )
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*{extension}"))
            self._walker = [
                w
                for w in walker
                if (HASH_DIVIDER in w and EXCEPT_FOLDER not in w)
                and any(
                    [
                        label == w.split("/")[2]
                        for label in self.local_label_mapping.keys()
                    ]
                )
            ]

        if sample_equally:
            walker_predict = [
                file
                for file in self._walker
                if file.split("/")[2] in labels_to_predict_mapping.keys()
            ]
            walker_unknown = [
                file for file in self._walker if file.split("/")[2] in unknown_labels
            ]
            random.shuffle(walker_unknown)
            walker_unknown = walker_unknown[: len(walker_predict)]
            self._walker = walker_predict + walker_unknown

    def get_metadata(self, n: int) -> Tuple[str, int, str, str, int]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            str:
                Path to the audio
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        fileid = self._walker[n]
        return _get_speechcommands_metadata(fileid, self._archive)

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Label
            str:
                Speaker ID
            int:
                Utterance number
        """
        metadata = self.get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        label = self.local_label_mapping[metadata[2]]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return (waveform, label)

    def get_label(self, n: int) -> str:
        return self.int_to_label[n]

    def local_to_global_label(self, local_label: int) -> int:
        str_label = self.int_to_label[local_label]
        return labels_to_predict_mapping.get(str_label, total_good_labels)

    def __len__(self) -> int:
        return len(self._walker)
