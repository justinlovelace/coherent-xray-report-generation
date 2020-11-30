import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import LSTM_Attn, CNN_Attn
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import sys
import json
from pprint import pprint

# Data parameters
data_folder = ''  # folder with data files saved by create_input_files.py
torch.set_num_threads(8)

# Model parameters
model_names = ['LSTM_ATTN', 'CNN_ATTN']
model_name = model_names[0]
emb_dim = 256  # dimension of word embeddings
lstm_dim = 128  # dimension of decoder RNN
cnn_dim = 64
cnn_kernels = [3, 5, 7, 9, 11]
dropout = 0.5
label_types = 3

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('ERROR GPU UNAVAILABLE')
    sys.exit()# cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 64  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
patience = 10
batch_size = 128
workers = 4  # for data-loading; right now, only 1 works with h5py
lr = 5e-4  # learning rate for decoder
best_f1 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none

if model_name == 'LSTM_ATTN':
    job_name = "{}_bs{}_lr{}_h{}".format(model_name, batch_size, lr, lstm_dim)
elif model_name == 'CNN_ATTN':
    job_name = "{}_bs{}_lr{}_f{}_k{}".format(model_name, batch_size, lr, cnn_dim, '_'.join([str(k) for k in cnn_kernels]))
else:
    print('invalid model')
    sys.exit()

data_name = os.path.join(data_folder, 'saved_chexpert_models', job_name)
if not os.path.exists(data_name):
    os.makedirs(data_name)
set_logger(os.path.join(data_name, 'train.log'))

logging.info(job_name)

def main():
    """
    Training and validation.
    """

    global best_f1, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    word_map = np.load(os.path.join(data_folder, 'word2ind.npy'), allow_pickle=True).item()

    # Custom dataloaders
    train_data = ReportDataset(data_folder, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ReportDataset(data_folder, 'val'), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Initialize / load checkpoint
    if checkpoint is None:
        datasetPath = os.path.join(data_folder, 'cxr_w2v.npy')
        emb_weights = np.load(datasetPath)
        if model_name == 'LSTM_ATTN':
            model = LSTM_Attn(emb_dim=emb_dim,
                           embed_weight=emb_weights,
                           hidden_size=lstm_dim)
        elif model_name == 'CNN_ATTN':
            model = CNN_Attn(emb_dim=emb_dim,
                            embed_weight=emb_weights,
                            filters=cnn_dim,
                            kernels=cnn_kernels)
        optimizer = torch.optim.Adam(params=model.parameters(),
                                    lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16, gamma=0.5)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_f1 = checkpoint['micro f1']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to GPU, if available
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8(5) consecutive epochs, and terminate training after 20(10)
        if epochs_since_improvement == 100:
            break
        # if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
        #     adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=epoch)
        # One epoch's validation
        metrics = validate(val_loader=val_loader,
                                    model=model,
                                    criterion=criterion)


        # Check if there was an improvement
        is_best = metrics['Micro Positive F1'] > best_f1
        best_f1 = max(metrics['Micro Positive F1'], best_f1)
        logging.info('Epoch: ' + str(epoch))
        logging.info('Recent Metrics: ' + str(json.dumps(metrics, indent=4, sort_keys=True)))
        logging.info('Recent F1: ' + str(metrics['Micro Positive F1']))
        logging.info('Best F1: ' + str(best_f1))
        print(job_name)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # scheduler.step()

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, model, optimizer, metrics['Micro Positive F1'], is_best)

        if epochs_since_improvement > patience:
            logging.info('Patience run out')
            logging.info('Best F1: ' + str(best_f1))
            print(job_name)
            break


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: loss layer
    :param optimizer: optimizer to update model's weights
    :param epoch: epoch number
    """

    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i, (labels, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        labels = labels.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores = model(caps, caplens)

        # Calculate loss
        loss = criterion(scores.view(-1, 3), labels.view(-1))

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        total_predictions = scores.size(0)*scores.size(1)
        acc = accuracy(scores.view(-1, 3).to('cpu'), labels.view(-1).to('cpu'))
        losses.update(loss.item(), total_predictions)
        accs.update(acc, total_predictions)
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          acc=accs))

def validate(val_loader, model, criterion):
    """
    Performs one epoch's training.
    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: loss layer
    """

    model.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    accs = AverageMeter()  # accuracy
    metrics = Metrics()

    start = time.time()

    # Batches
    with torch.no_grad():
        for i, (labels, caps, caplens) in enumerate(val_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            labels = labels.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores = model(caps, caplens)

            # Calculate loss
            loss = criterion(scores.view(-1, 3), labels.view(-1))

            # Keep track of metrics
            total_predictions = scores.size(0)*scores.size(1)
            acc = accuracy(scores.view(-1, 3).to('cpu'), labels.view(-1).to('cpu'))
            losses.update(loss.item(), total_predictions)
            accs.update(acc, total_predictions)
            batch_time.update(time.time() - start)
            metrics.update(scores.to('cpu'), labels.to('cpu'))

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, acc=accs))

    metrics_dict = metrics.calculate_metrics()
    print(
        '\n * LOSS - {loss.avg:.3f}\n'.format(
            loss=losses))

    return metrics_dict


if __name__ == '__main__':
    main()