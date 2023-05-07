from audio_data import (
    AudioDataModule,
    MFCC_transform,
    wave2vec_transform,
    spectrogram_transform,
)
from myspeechcommands import unknown_labels
from models import LSTMDenseClassifier, LSTMGRUMODEL, DenseClassifier
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import pprint

seed_everything(7)

predicted = [
    "yes",
    "no",
    "unknown",
]


class MyPrintingCallback(Callback):
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     preds = torch.cat([tmp["preds"] for tmp in pl_module.validation_step_outputs])
    #     targets = torch.cat(
    #         [tmp["target"] for tmp in pl_module.validation_step_outputs]
    #     )
    #     confusion_matrix = pl_module.confusion_matrix(preds, targets)
    #     df_cm = pd.DataFrame(
    #         confusion_matrix.cpu().numpy(),
    #         index=predicted,
    #         columns=predicted,
    #     )
    #     plt.figure(figsize=(10, 7))
    #     fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
    #     plt.close(fig_)

    #     trainer.logger.experiment.add_figure(
    #         "Confusion matrix", fig_, pl_module.current_epoch
    #     )
    #     pl_module.validation_step_outputs = []

    def on_test_end(self, trainer, pl_module):
        preds = torch.cat([tmp["preds"] for tmp in pl_module.test_step_outputs])
        targets = torch.cat([tmp["target"] for tmp in pl_module.test_step_outputs])
        confusion_matrix = pl_module.confusion_matrix(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(),
            index=predicted,
            columns=predicted,
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)

        trainer.logger.experiment.add_figure("Confusion matrix", fig_)
        pl_module.test_step_outputs = []


torch.set_float32_matmul_precision("medium")

data_module = AudioDataModule(
    data_dir="./",
    batch_size=1,
    data_transform=spectrogram_transform(),
    # data_transform=MFCC_transform(),
    label_subset=predicted.copy(),
)

data_module.prepare_data()
dims = data_module.get_data_dimensions()

# moment_dict = {}
# for label in predicted:
#     moment_dict[label] = []
#
# # x = data_module.speechcommands_train[j * 5000 + i]
# for x in data_module.speechcommands_train:
#     spectrogram = x[0]
#     label = x[1]
#     label = data_module.speechcommands_train.get_label(label)
#     moments = cv2.moments(spectrogram)
#     huMoments = cv2.HuMoments(moments)
#     # log transfrom
#     for i in range(0, 7):
#         if huMoments[i] != 0:
#             huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i]))
#     moment_dict[label].append(huMoments)
#     # print(label)
#     # print(huMoments)
#     # plt.figure(figsize=(10, 7))
#     # plt.imshow(spectrogram)
#     # plt.title(f"Label: {label}")
#     # plt.show()
#
# # get average hu moments for each label
#
# for label in moment_dict:
#     moment_dict[label] = np.array(moment_dict[label]).mean(axis=0)
#
# pprint.pprint(moment_dict)
#
#
sequence_length = dims[0]
input_size = dims[1]

print(f"Sequence length: {sequence_length}, input size: {input_size}")

# model = LSTMDenseClassifier(
#     input_size=input_size,
#     hidden_size=128,
#     num_layers=1,
#     num_classes=3,
#     classes=predicted,
# )

model = DenseClassifier(
    input_size=7,
    hidden_size=128,
    num_classes=3,
    classes=predicted,
)

trainer = Trainer(
    max_epochs=150,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else 0,
    logger=TensorBoardLogger("lightning_logs", name="dense_test_yes_no"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        MyPrintingCallback(),
        EarlyStopping(monitor="val_loss", patience=15, verbose=True),
    ],
)

trainer.fit(model, datamodule=data_module)
x = trainer.test(model, datamodule=data_module)
print(x)
