from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform
from myspeechcommands import unknown_labels
from models import LSTMDenseClassifier, LSTMGRUMODEL
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

seed_everything(7)

predicted = ["no", "up", "unknown"]


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
    batch_size=128,
    data_transform=MFCC_transform(),
    label_subset=predicted.copy(),
)

data_module.prepare_data()
dims = data_module.get_data_dimensions()
sequence_length = dims[0]
input_size = dims[1]

print(f"Sequence length: {sequence_length}, input size: {input_size}")

model = LSTMDenseClassifier(
    input_size=input_size,
    hidden_size=128,
    num_layers=1,
    num_classes=3,
)

trainer = Trainer(
    max_epochs=450,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else 0,
    logger=TensorBoardLogger("lightning_logs", name="lstm_dense_no_up"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        MyPrintingCallback(),
    ],
)

trainer.fit(model, datamodule=data_module)
x = trainer.test(model, datamodule=data_module)
print(x)
