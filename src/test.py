from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform
from models import LSTMDenseClassifier
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

data_module = AudioDataModule(
    data_dir="./", batch_size=64, data_transform=MFCC_transform()
)
data_module.prepare_data()

print(data_module.get_data_dimensions())

# for batch in data_module.train_dataloader():
#     print(batch[0].shape)
#     print(batch[1].shape)

model = LSTMDenseClassifier(
    input_size=32, hidden_size=128, num_classes=10, sequence_length=100
)

trainer = Trainer(
    max_epochs=30,
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
