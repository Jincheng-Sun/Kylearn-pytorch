from utils.embeddings import one_hot_embedding
import torch


def performance_multi(logits, real):
    logits = logits.max(1)[1]
    real = real.contiguous().view(-1)
    non_pad_mask = real.ne(0)
    n_correct = logits.eq(real)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


def performance_regression(logits, real, threshold):
    ''' Apply label smoothing if needed '''

    logits = logits.ge(threshold)
    real = real.contiguous().view(-1)
    non_pad_mask = real.ne(0)
    n_correct = logits.eq(real)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return n_correct


def accuracy(pred, real):
    n = real.shape[0]
    pred = pred.argmax(dim=-1).view(n, -1)
    real = real.view(n, -1)
    return (pred == real).float().mean()


def accuracy_threshold(logits, real, threshold=0.5):
    logits = logits.sigmoid()
    return ((logits > threshold).byte() == real.byte()).float().mean()


def confusion_matrix(pred, real, threshold=None):
    '''
        For binary or multi class classification
        If pred is probabilities, convert to binary form
    '''
    dim = real.shape[-1]
    if dim == 1:
        if threshold != None:
            pred = (pred > threshold).byte()
            pred = one_hot_embedding(pred, 2)
            real = one_hot_embedding(real, 2)
        else:
            pred = one_hot_embedding(pred, real.max(-1))
            real = one_hot_embedding(real, real.max(-1))

    cm = torch.bmm(real.t(), pred)
    return cm


def precision_racall(pred, real, threshold=None, average='macro'):
    n_sample = real.shape[0]
    dim = real.shape[-1]
    if dim == 1:
        if threshold != None:
            pred = (pred > threshold).byte()
            pred = one_hot_embedding(pred, 2)
            real = one_hot_embedding(real, 2)
        else:
            pred = one_hot_embedding(pred, real.max(-1))
            real = one_hot_embedding(real, real.max(-1))

    tp = (pred & real).sum(0).float()
    fp = (pred - tp).sum(0).float()
    fn = (real - tp).sum(0).float()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if threshold != None:
        return precision[0], threshold[0], precision[0], threshold[0]

    else:
        if average == 'macro':
            precision_avg = precision.mean()
            recall_avg = precision.mean()

        else:
            precision_avg = tp.sum(0).sum() / n_sample
            recall_avg = precision_avg

    return precision, recall, precision_avg, recall_avg