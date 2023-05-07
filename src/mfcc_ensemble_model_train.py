from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform
from myspeechcommands import labels_to_predict_mapping
from models import LSTMDenseClassifier, LSTMGRUMODEL
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, Callback, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from tqdm import tqdm


torch.set_float32_matmul_precision("medium")

base_labels = list(labels_to_predict_mapping.keys())
base_labels.remove("_background_noise_")

all_variants = combinations(base_labels, 2)
for i, variant in enumerate(all_variants):
    predicted = list(variant)
    predicted.append("unknown")
    print(f"Predicting {predicted}")
    print(f"Model {i+1}/{len(list(all_variants))}")
    seed_everything(7)
    
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

    # sanity check for labels remapping
    datasets = [data_module.speechcommands_test, data_module.speechcommands_train, data_module.speechcommands_val]
    for dataset in datasets:
        for i, l in dataset.int_to_label.items():
            if predicted.index(l) != i:
                print(f'Dataset mapping: {dataset.int_to_label}')
                print(f"Label {l} should be {i} but is {predicted.index(l)}")
                raise ValueError("Labels remapping is not correct")
    
    print(f"Sequence length: {sequence_length}, input size: {input_size}")
    
    model = LSTMDenseClassifier(
        input_size=input_size,
        hidden_size=128,
        num_layers=1,
        num_classes=3,
        classes=predicted.copy(),
    )
    
    trainer = Trainer(
        max_epochs=150,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else 0,
        logger=TensorBoardLogger("lightning_logs", name=f'lstm_mfcc_{predicted[0]}_{predicted[1]}_shuffle_test'),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=0),
            MyPrintingCallback(),
            EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    
    trainer.fit(model, datamodule=data_module)
    x = trainer.test(model, datamodule=data_module)
    print(x)
    