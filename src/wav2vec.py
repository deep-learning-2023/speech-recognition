from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform, pad_to_max_length_collator, discard_if_unknown
from myspeechcommands import unknown_labels, without_noise
from models import LSTMDenseClassifier, LSTMGRUMODEL
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from time import sleep

seeds = [7, 71, 717]

p = product([32, 64, 128], [1])

for hidden_size, num_layers in p:
    for seed in seeds:
    
        seed_everything(seed)
        
        predicted = without_noise.copy()
        
        
        class MyPrintingCallback(Callback):
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
            # collate_fn=pad_to_max_length_collator,
            # data_transform=discard_if_unknown,
            label_subset=predicted.copy(),
            wav2vec=True
        )
        
        data_module.prepare_data()
        dims = data_module.get_data_dimensions()
        sequence_length = dims[0]
        input_size = dims[1]
        
        print(f"Sequence length: {sequence_length}, input size: {input_size}")
        
        model = LSTMDenseClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(predicted),
            classes=predicted,
        )
    
        trainer = Trainer(
            max_epochs=150,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else 0,
            logger=TensorBoardLogger("lightning_logs", name=f"wav2vec_lstm_notrim"),
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
#                TQDMProgressBar(refresh_rate=10),
                MyPrintingCallback(),
                EarlyStopping(monitor="val_loss", patience=10, verbose=True),
            ],
        )
        
        trainer.fit(model, datamodule=data_module)
        x = trainer.test(model, datamodule=data_module)
        print(x)
        sleep(1)
    