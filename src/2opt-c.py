import os
import csv
from itertools import combinations
from math import ceil
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from audio_data import AudioDataModule, MFCC_transform
import pandas as pd

from models import LSTMDenseClassifier

from eval_acc import softvote, hardvote, softvote_pred, hardvote_pred
from myspeechcommands import labels_to_predict_mapping

from sklearn.metrics import confusion_matrix
from torchmetrics.functional import accuracy

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seed_everything(71)

BATCH_SIZE = 256
def VOTE(preds, y):
    # filter entries with highest probability of the last class
    return softvote(preds, y)

# list of models log directories (it is assumed that each model has its own directory)
models_dirs = [
"lightning_logs/lstm_mfcc_down_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_down_left_shuffle_test_final",
"lightning_logs/lstm_mfcc_down_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_down_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_down_right_shuffle_test_final",
"lightning_logs/lstm_mfcc_down_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_left_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_left_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_left_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_left_right_shuffle_test_final",
"lightning_logs/lstm_mfcc_left_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_down_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_left_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_right_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_no_up_shuffle_test_final",
"lightning_logs/lstm_mfcc_off_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_off_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_on_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_on_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_on_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_right_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_right_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_right_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_right_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_stop_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_down_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_left_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_right_shuffle_test_final",
"lightning_logs/lstm_mfcc_up_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_down_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_go_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_left_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_no_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_off_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_on_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_right_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_stop_shuffle_test_final",
"lightning_logs/lstm_mfcc_yes_up_shuffle_test_final"
]

print(f'Evaluating {len(models_dirs)} models')
models_dirs = list(map(lambda x: os.path.join(x, "version_0"), models_dirs))
models_dirs = {tuple(filename.split("_")[3:5]): filename for filename in models_dirs}

global_int_to_str_mapping = {v: k for k, v in labels_to_predict_mapping.items()}
global_int_to_str_mapping[10] = "unknown"
global_str_to_int_mapping = {v: k for k, v in global_int_to_str_mapping.items()}

def map_to_global(label_list, y):
    # label list: 2 labels being strings corresponding to global labels
    # return 11 elem tensor with associated global labels
    y = y.tolist()
    new_y = []
    idx1 = global_str_to_int_mapping[label_list[0]]
    idx2 = global_str_to_int_mapping[label_list[1]]
    for y_elem in y:
        zero_tensor = torch.zeros(11)
        if y_elem[2] < y_elem[0] or y_elem[2] < y_elem[1]:
            zero_tensor[idx1] = y_elem[0]
            zero_tensor[idx2] = y_elem[1]
            zero_tensor[10] = y_elem[2]
        new_y.append(zero_tensor)

    return torch.stack(new_y)

# load models and scores, tuples (dir, filename, score) from csv
models = {}
print("Loading models...")
for predicting_classes, model_dir in models_dirs.items():
    # find .ckpt file
    file = os.listdir(os.path.join(model_dir, "checkpoints"))[0]
    model = LSTMDenseClassifier.load_from_checkpoint(os.path.join(model_dir, "checkpoints", file))
    model.eval()
    model.to('cuda')
    models[predicting_classes] = model

data_module = AudioDataModule(
        data_dir="./",
        batch_size=BATCH_SIZE,
        data_transform=MFCC_transform()
)
data_module.prepare_data()

targets = []
model_predictions = {c: [] for c in models.keys()}

print("Computing predictions...")
for batch in data_module.train_dataloader():
    x, y = batch
    # drop x, y where y == 10
    x = x[y != 10]
    y = y[y != 10]
    targets.append(y)
    for c, model in models.items():
        with torch.no_grad():
            preds = model(x.to('cuda'))
            # apply softmax
            preds = torch.nn.functional.softmax(preds, dim=1)
            model_predictions[c].append(preds)

# concatenate all targets
targets = torch.cat(targets)

models_acc = {}

print("Computing accuracy...")
# remap each model prediction to global labels
for c, preds in tqdm(model_predictions.items()):
    # concatenate all predictions
    preds = torch.cat(preds)
    model_predictions[c] = map_to_global(c, preds)
    models_acc[c] = VOTE([model_predictions[c]], targets)

# sort models by accuracy
xd = sorted(models_acc.items(), key=lambda item: item[1])
xd = list(map(lambda x: x[0], xd))

model_predictions = {idx: model_predictions[k] for idx, k in enumerate(xd)}


def add(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best
    for i, model in model_predictions.items():
        if i not in c_best:
            new_ensemble = c_best.copy()
            new_ensemble[i] = model
            acc = VOTE(list(new_ensemble.values()), targets)
            if acc > local_best_acc:
                local_best_acc = acc
                local_best_C = new_ensemble
                break

    return local_best_C, local_best_acc

def remove(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best
    
    if len(c_best) == 1:
        return local_best_C, local_best_acc
    
    for i, _ in c_best.items():
        new_ensemble = c_best.copy()
        new_ensemble.pop(i)
        acc = VOTE(list(new_ensemble.values()), targets)
        if acc > local_best_acc:
            local_best_acc = acc
            local_best_C = new_ensemble
            break     
    
    return local_best_C, local_best_acc



def swap(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best
    for i, model in model_predictions.items():
        if i not in c_best:
            for j, _ in c_best.items():
                new_ensemble = c_best.copy()
                new_ensemble.pop(j)
                new_ensemble[i] = model
                acc = VOTE(list(new_ensemble.values()), targets)
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_C = new_ensemble
                    break
    
    return local_best_C, local_best_acc

def addTwo(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best
    for i, model in model_predictions.items():
        if i not in c_best:
            for j, model2 in model_predictions.items():
                if j not in c_best and j != i:
                    new_ensemble = c_best.copy()
                    new_ensemble[i] = model
                    new_ensemble[j] = model2
                    acc = VOTE(list(new_ensemble.values()), targets)
                    if acc > local_best_acc:
                        local_best_acc = acc
                        local_best_C = new_ensemble
                        break
    
    return local_best_C, local_best_acc

def removeTwo(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best

    if len(c_best) <= 2:
        return local_best_C, local_best_acc

    for i, _ in c_best.items():
        for j, _ in c_best.items():
            if j != i:
                new_ensemble = c_best.copy()
                new_ensemble.pop(j)
                new_ensemble.pop(i)
                acc = VOTE(list(new_ensemble.values()), targets)
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_C = new_ensemble
                    break     
    
    return local_best_C, local_best_acc

def addAndSwap(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best

    for (to_add1, model1), (to_add2, model2) in combinations(model_predictions.items(), 2):
        if to_add1 not in c_best and to_add2 not in c_best:
            for to_delete, _ in c_best.items():
                new_ensemble = c_best.copy()
                new_ensemble.pop(to_delete)
                new_ensemble[to_add1] = model1
                new_ensemble[to_add2] = model2
                acc = VOTE(list(new_ensemble.values()), targets)
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_C = new_ensemble
                    break
    
    return local_best_C, local_best_acc

def removeAndSwap(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best
    if len(c_best) <= 2:
        return local_best_C, local_best_acc

    for (to_delete1, _), (to_delete2, _) in combinations(c_best.items(), 2):
        for to_add, model_to_add in model_predictions.items():
            if to_add not in c_best:
                new_ensemble = c_best.copy()
                new_ensemble.pop(to_delete1)
                new_ensemble.pop(to_delete2)
                new_ensemble[to_add] = model_to_add
                acc = VOTE(list(new_ensemble.values()), targets)
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_C = new_ensemble
                    break
    
    return local_best_C, local_best_acc

def swapTwice(c_best: dict, acc_best):
    local_best_acc = acc_best
    local_best_C = c_best

    if len(c_best) == 1:
        return local_best_C, local_best_acc
    
    for (to_delete1, _), (to_delete2, _) in combinations(c_best.items(), 2):
        for (to_add1, model1), (to_add2, model2) in combinations(model_predictions.items(), 2):
            if to_add1 not in c_best and to_add2 not in c_best:
                new_ensemble = c_best.copy()
                new_ensemble.pop(to_delete1)
                new_ensemble.pop(to_delete2)
                new_ensemble[to_add1] = model1
                new_ensemble[to_add2] = model2
                acc = VOTE(list(new_ensemble.values()), targets)
                if acc > local_best_acc:
                    local_best_acc = acc
                    local_best_C = new_ensemble
                    break
    return local_best_C, local_best_acc
    

def jumper(c_best: dict, acc_curr, acc_best, local_c, f):
    if acc_best >= acc_curr:
        return f(c_best, acc_best)
    else:
        return local_c, acc_curr
    

c_best = {0: model_predictions[0]}
acc_best = VOTE(list(c_best.values()), targets)

while True:
    acc_curr = acc_best
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, c_best, lambda x,y: add(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: remove(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: swap(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: addTwo(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: removeTwo(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: addAndSwap(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: removeAndSwap(x, y))
    local_best_C, acc_curr = jumper(c_best, acc_curr, acc_best, local_best_C, lambda x,y: swapTwice(x, y))
    if acc_curr > acc_best:
        c_best = local_best_C
        acc_best = acc_curr
        print(acc_best)
        print(f'Selected models: {list(c_best.keys())}')
    else:
        break

#df_class_columns = [f'Class {i}' for i in range(10)]
#results = pd.DataFrame(columns=['model', 'accuracy']+df_class_columns)
for i in c_best:
    #y_hat = model_predictions[i].argmax(dim=1)
    #print(y_hat.shape)
    #conf_mat = confusion_matrix(targets, y_hat, normalize='true').diagonal()
    model_name = xd[i]
    print(model_name)
#    results.loc[i] = [model_name] + list(conf_mat)

ensemble_predictions = list(c_best.values())
y_hat = softvote_pred(ensemble_predictions)
conf_mat = confusion_matrix(targets, y_hat, normalize='true')
# save confusion matrix as pdf
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, ax=ax, cmap='Blues', fmt='.2f')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels([global_int_to_str_mapping[i] for i in range(11)])
ax.yaxis.set_ticklabels([global_int_to_str_mapping[i] for i in range(11)])
plt.savefig('confusion_matrix.pdf')

# calculate precision, recall
precision = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
recall = np.diag(conf_mat) / np.sum(conf_mat, axis=1)
print('precision:', precision)
print('recall:', recall)

#results.loc['ensemble'] = ['ensemble', acc_best.item()] + list(conf_mat)


#results.to_csv('ensemble_results.csv', index=False)