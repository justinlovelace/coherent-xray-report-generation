import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import LSTM_Attn, CNN_Attn
from utils import *
from datasets import *
import numpy as np
import os
import sys
import json
import csv

# Data parameters
data_folder = ''  # folder with data files saved by create_input_files.py

# Model parameters
model_names = ['LSTM_ATTN', 'CNN_ATTN']
model_name = model_names[1]
emb_dim = 256  # dimension of word embeddings
lstm_dim = 128  # dimension of decoder RNN
cnn_dim = 64
cnn_kernels = [3, 5, 7, 9]
dropout = 0.5
label_types = 3
split = 'test'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('ERROR GPU UNAVAILABLE')
    sys.exit()# cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 64  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation F1
patience = 10
batch_size = 128
workers = 4  # for data-loading
lr = 5e-4  # learning rate for decoder
best_f1 = 0.  # F1 score right now
print_freq = 5  # print training/validation stats every __ batches
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

set_logger(os.path.join(data_name, split + '.log'))

logging.info(job_name)
logging.info(split)

checkpoint=os.path.join(data_name, 'BEST_checkpoint.pth.tar')


# Load model
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

# Load word map (word2ix)
word_map = np.load(os.path.join(data_folder, 'word2ind.npy'), allow_pickle=True).item()
ind2word = np.load(os.path.join(data_folder, 'ind2word.npy'), allow_pickle=True).item()

vocab_size = len(word_map)-2

# DataLoader
loader = torch.utils.data.DataLoader(
        ReportDataset(data_folder,  split), batch_size=512, shuffle=False, num_workers=1, pin_memory=True)


def evaluate(data_loader, model, criterion):
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
        for i, (labels, caps, caplens) in enumerate(data_loader):
            data_time.update(time.time() - start)

            # Move to GPU, if available
            labels = labels.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores = model(caps, caplens)
            # print(scores.view(-1, 3).size())
            # print(labels.view(-1, 1).size())
            # sys.exit()

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
                      'Top-5 Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(data_loader), batch_time=batch_time,
                                                                                loss=losses, acc=accs))

    metrics_dict = metrics.calculate_metrics()
    print(
        '\n * LOSS - {loss.avg:.3f}\n'.format(
            loss=losses))

    return metrics_dict


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(job_name)
    metrics_dict = evaluate(loader, model, criterion)
    logging.info(str(json.dumps(metrics_dict, indent=4, sort_keys=True)))

    csv_columns = list(metrics_dict.keys())
    csv_file = os.path.join(data_name, split + '_chexpert.csv')
    print(csv_file)
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(metrics_dict)