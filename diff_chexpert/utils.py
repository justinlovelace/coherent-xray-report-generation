import os
import numpy as np
import json
import torch
import logging
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import sklearn.metrics as sk
import sys

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer,
                    micro_f1, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param decoder: decoder model
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'micro f1': micro_f1,
             'model': model,
             'optimizer': optimizer}
    filename = os.path.join(data_name, 'checkpoint.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, os.path.join(data_name, 'BEST_checkpoint.pth.tar'))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    scores = torch.argmax(scores, dim=1)
    return sk.accuracy_score(scores, targets)

class Metrics(object):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.cond_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other', 'Fracture', 'Support Devices']
        for i in range(14):
            self.y_true.append([])
            self.y_pred.append([])

    def update(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        y_pred = torch.argmax(y_pred, dim=2)
        # print(y_pred.size())

        for i in range(len(self.cond_names)):
            self.y_true[i].append(y_true[:, i])
            self.y_pred[i].append(y_pred[:, i])


    def calculate_metrics(self):
        metrics = {}

        # Compute metrics for each condition
        for i in range(len(self.cond_names)):
            y_true = torch.cat(self.y_true[i])
            y_pred = torch.cat(self.y_pred[i])

            metrics['Positive Precision ' + self.cond_names[i]] = list(sk.precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive Recall ' + self.cond_names[i]] = list(sk.recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive F1 ' + self.cond_names[i]] = list(sk.f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]

            metrics['Uncertain Precision ' + self.cond_names[i]] = \
            list(sk.precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain Recall ' + self.cond_names[i]] = \
            list(sk.recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain F1 ' + self.cond_names[i]] = list(sk.f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[
                0]

            # metrics['Micro Precision '+self.cond_names[i]] = sk.precision_score(y_true, y_pred, average='micro')
            # metrics['Micro Recall ' + self.cond_names[i]] = sk.recall_score(y_true, y_pred, average='micro')
            # metrics['Micro F1 ' + self.cond_names[i]] = sk.f1_score(y_true, y_pred, average='micro')

        # Compute global metrics
        master_y_true = torch.cat([inner for outer in self.y_true for inner in outer])
        master_y_pred = torch.cat([inner for outer in self.y_pred for inner in outer])

        metrics['Micro Positive Precision'] = list(sk.precision_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive Recall'] = list(sk.recall_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive F1'] = list(sk.f1_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]

        metrics['Micro Uncertain Precision'] = \
        list(sk.precision_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain Recall'] = \
        list(sk.recall_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain F1'] = list(sk.f1_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]

        return metrics


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
