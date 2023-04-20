from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform
from models import LSTMDenseClassifier, LSTMGRUMODEL
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision('medium')

data_module = AudioDataModule(
    data_dir="./", batch_size=128, data_transform=MFCC_transform(), label_subset=["yes", "no", "up", "down", "left", "right"]
)

data_module.prepare_data()
dims = data_module.get_data_dimensions()
sequence_length = dims[0]
input_size = dims[1]

print(f"Sequence length: {sequence_length}, input size: {input_size}")

# model = LSTMDenseClassifier(
#     input_size=input_size, hidden_size=128, num_layers = 2, num_classes=6
# )

model = LSTMGRUMODEL(
    input_size=input_size, hidden_size=128, num_layers = 2, num_classes=6
)

trainer = Trainer(
    max_epochs=300,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else 0,
    logger=CSVLogger(save_dir="logs/"),
    callbacks=[
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ],
)

trainer.fit(model, datamodule=data_module)
trainer.test(model, datamodule=data_module)
