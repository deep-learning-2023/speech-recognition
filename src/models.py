import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
import pytorch_lightning as pl


class DenseClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes, classes: list[str]):
        super().__init__()
        super().save_hyperparameters()
        self.classes = classes
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.validation_step_outputs.append({"preds": y_hat, "target": y})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_precision", self.precision(y_hat, y))
        self.log("test_recall", self.recall(y_hat, y))
        self.test_step_outputs.append({"preds": y_hat, "target": y})


class CNNDenseClassifier(pl.LightningModule):
    def __init__(self, num_classes, classes: list[str]):
        super().__init__()
        super().save_hyperparameters()
        self.classes = classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=16)
        self.fc1 = nn.Linear(960, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.validation_step_outputs.append({"preds": y_hat, "target": y})

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_precision", self.precision(y_hat, y))
        self.log("test_recall", self.recall(y_hat, y))
        self.test_step_outputs.append({"preds": y_hat, "target": y})


class LSTMDenseClassifier(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, classes: list[str]
    ):
        super().__init__()
        super().save_hyperparameters()
        self.num_layers = num_layers
        self.classes = classes

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=self.num_layers, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.validation_step_outputs.append({"preds": y_hat, "target": y})
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.test_step_outputs.append({"preds": y_hat, "target": y})
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # precision = pl.metrics.functional.precision(y_hat, y)
        # recall = pl.metrics.functional.recall(y_hat, y)
        acc = self.accuracy(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)


# class ValidationCallback(Callback):
#     def on_validation_epoch_end(self, trainer, pl_module):
#         preds = torch.cat([tmp["preds"] for tmp in pl_module.validation_step_outputs])
#         targets = torch.cat(
#             [tmp["target"] for tmp in pl_module.validation_step_outputs]
#         )
#         confusion_matrix = pl_module.confusion_matrix(preds, targets)
#         df_cm = pd.DataFrame(
#             confusion_matrix.numpy(), index=range(10), columns=range(10)
#         )
#         plt.figure(figsize=(10, 7))
#         fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
#         plt.close(fig_)
#
#         trainer.logger.experiment.add_figure(
#             "Confusion matrix", fig_, pl_module.current_epoch
#         )
#
#
# class TestCallback(Callback):
#     def on_test_end(self, trainer, pl_module):
#         preds = torch.cat([tmp["preds"] for tmp in pl_module.test_step_outputs])
#         targets = torch.cat([tmp["target"] for tmp in pl_module.test_step_outputs])
#         confusion_matrix = pl_module.confusion_matrix(preds, targets)
#         df_cm = pd.DataFrame(
#             confusion_matrix.numpy(), index=range(10), columns=range(10)
#         )
#         plt.figure(figsize=(10, 7))
#         fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
#         plt.close(fig_)
#
#         trainer.logger.experiment.add_figure("Confusion matrix", fig_)


class LSTMGRUMODEL(pl.LightningModule):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, classes: list[str]
    ):
        super().__init__()
        super().save_hyperparameters()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.classes = classes

        self.log("model_class", 2)

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gru = nn.GRU(
            hidden_size * 2,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.precision = Precision(task="multiclass", num_classes=num_classes)
        self.recall = Recall(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        gru_out, _ = self.gru(lstm_out)

        out = self.fc(gru_out[:, -1, :])
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        acc = self.accuracy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.validation_step_outputs.append({"preds": y_hat, "target": y})
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = self.accuracy(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.test_step_outputs.append({"preds": y_hat, "target": y})
        # acc = (y_hat.argmax(dim=1) == y).float().mean()
        # precision = pl.metrics.functional.precision(y_hat, y)
        # recall = pl.metrics.functional.recall(y_hat, y)
        acc = self.accuracy(y_hat, y)
        precision = self.precision(y_hat, y)
        recall = self.recall(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_precision", precision)
        self.log("test_recall", recall)
