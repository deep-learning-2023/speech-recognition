import torch
from torchmetrics.functional import accuracy

def eval_acc(dataloader, models_in_ensemble):

    models_in_ensemble = list(models_in_ensemble)

    hard_voting_acc = 0
    soft_voting_acc = 0
    
    for batch in dataloader:
        x, y = batch
        batch_size = x.shape[0]
        preds = [model(x) for model in models_in_ensemble]
        
        hard_voting_acc += hardvote(preds, y) * batch_size
    
        soft_voting_acc += softvote(preds, y) * batch_size
        
    
    hard_voting_acc /= len(dataloader.dataset)
    soft_voting_acc /= len(dataloader.dataset)
    
    return hard_voting_acc, soft_voting_acc

def hardvote_pred(preds):
    stacked = torch.stack(preds, dim=1)
    # max = torch.argmax(stacked, dim=2)
    # zeros = torch.zeros(stacked.shape)
    
    # for z in range(stacked.shape[0]):
    #     for j in range(stacked.shape[1]):
    #         zeros[z, j, max[z, j]] = 1
    
    # voted = torch.sum(zeros, dim=1)
    
    # y_hat = torch.argmax(voted, dim=1)
    y_hat = torch.mode(stacked, dim=1)[0]
    return y_hat

def hardvote(preds, y):
    y_hat = hardvote_pred(preds)
    return accuracy(y_hat, y, 'multiclass', num_classes=11)

def softvote_pred(preds):
    y_hat = torch.mean(torch.stack(preds, dim=1), dim=1)
    y_hat = torch.argmax(y_hat, dim=1)
    return y_hat

def softvote(preds, y):
    y_hat = softvote_pred(preds)
    return accuracy(y_hat, y, 'multiclass', num_classes=11)
