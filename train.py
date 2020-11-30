import time
import logging
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
from models import TransformerModel, TransformerGumbelModel
from diff_chexpert.models import LSTM_Attn
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import os
import sys

# Data parameters
data_folder = '/home/ugrads/j/justinlovelace/MIMIC/cxr/data'  # folder with data files saved by create_input_files.py

# Model parameters
emb_dim = 256  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print('ERROR GPU UNAVAILABLE')
    sys.exit()
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
model_names = ['TX', 'TXGB']
model_name = model_names[0]
start_epoch = 0
if 'TXGB' in model_name:
    epochs = 8  # number of epochs to train for (if early stopping is not triggered)
else:
    epochs = 64  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 8  # for data-loading; right now, only 1 works with h5py
decoder_lr = 5e-5  # learning rate for decoder
grad_clip = 1.  # clip gradients at an absolute value of
best_bleu4 = 0.  # BLEU-4 score right now
best_f1 = 0.  # F1 score right now
print_freq = 250  # print training/validation stats every __ batches
checkpoint = False  # path to checkpoint, None if none
chexpert_model_name = 'lstm'
if chexpert_model_name == 'lstm':
    chexpert_path = os.path.join(data_folder, 'saved_chexpert_models', 'LSTM_ATTN_bs128_lr0.0005_h128', 'BEST_checkpoint.pth.tar')  # path to checkpoint, None if none
else:
    chexpert_path = os.path.join(data_folder, 'saved_chexpert_models', 'LSTM_ATTN_bs128_lr0.0005_h128', 'BEST_checkpoint.pth.tar')  # path to checkpoint, None if none
chexpert_c = .9
temperature = 1
beta = 1
aux_weight = .5
fine_tune_lr = 1e-5  # learning rate for decoder

nhead = 8
d_model = 256
dim_feedforward = 4096
num_encoder_layers = 1
num_decoder_layers = 6

if model_name == 'TX':
    job_name = "{}_bs{}_lr{}_nhead{}_dmodel{}_dimff{}_enclayers{}_declayers{}_clip{}".format(model_name, batch_size, decoder_lr, nhead,
                                                                             d_model, dim_feedforward,
                                                                             num_encoder_layers, num_decoder_layers, grad_clip)
elif 'TXGB' in model_name:
    job_name = "{}_bs{}_lr{}_finetunelr{}_nhead{}_dmodel{}_dimff{}_enclayers{}_declayers{}_chexpert{}_temp{}_beta{}".format(model_name, batch_size, decoder_lr, fine_tune_lr,
                                                                             nhead, d_model, dim_feedforward,
                                                                             num_encoder_layers, num_decoder_layers, chexpert_c, temperature, beta)
    tx_name = "{}_bs{}_lr{}_nhead{}_dmodel{}_dimff{}_enclayers{}_declayers{}".format('TX', batch_size,
                                                                                      decoder_lr, nhead,
                                                                                      d_model, dim_feedforward,
                                                                                      num_encoder_layers,
                                                                                      num_decoder_layers)
    checkpoint_path = os.path.join(data_folder, 'saved_models', tx_name, 'BEST_checkpoint.pth.tar')

data_name = os.path.join(data_folder, 'saved_models', job_name)
if not os.path.exists(data_name):
    os.makedirs(data_name)
set_logger(os.path.join(data_name, 'train.log'))

logging.info(job_name)

def main():
    """
    Training and validation.
    """

    global best_bleu4, best_f1, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    word_map = np.load(os.path.join(data_folder, 'word2ind.npy'), allow_pickle=True).item()

    # Custom dataloaders
    train_data = ReportDataset(data_folder, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ReportDataset(data_folder, 'val'), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Initialize / load checkpoint
    if not checkpoint:
        datasetPath = os.path.join(data_folder, 'cxr_w2v.npy')
        emb_weights = np.load(datasetPath)
        if model_name == 'TX':
            decoder = TransformerModel(embed_weight=emb_weights,
                                       d_model=d_model,
                                       nhead=nhead,
                                       dim_feedforward=dim_feedforward,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       seq_len=train_data.pad_len)
        elif model_name == 'TXGB':
            checkpoint = torch.load(checkpoint_path)
            transformer = checkpoint['decoder']
            checkpoint = torch.load(chexpert_path)
            chexpert = checkpoint['model']
            print('Successfully loaded both models')
            decoder = TransformerGumbelModel(embed_weight=emb_weights,
                                            transformer=transformer,
                                            chexpert=chexpert,
                                            temperature=temperature,
                                            beta=beta)
        if 'GB' in model_name:
            decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),
                                                 lr=fine_tune_lr)
        else:
            decoder_optimizer = torch.optim.Adam(params=decoder.parameters(),
                                                 lr=decoder_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=16, gamma=0.5)
    else:
        print('wrong path')
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)



    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8(5) consecutive epochs, and terminate training after 20(10)
        if epochs_since_improvement == 100:
            break

        # One epoch's training
        if model_name == 'TXGB':
            fine_tune(train_loader=train_loader,
                  decoder=decoder,
                  criterion=criterion,
                  decoder_optimizer=decoder_optimizer,
                  epoch=epoch)
        else:
            train(train_loader=train_loader,
                  decoder=decoder,
                  criterion=criterion,
                  decoder_optimizer=decoder_optimizer,
                  epoch=epoch)
        # One epoch's validation
        if 'GB' in model_name:
            recent_bleu4, recent_f1 = validate(val_loader=val_loader,
                                        decoder=decoder,
                                        criterion=criterion)
        else:
            recent_bleu4 = validate(val_loader=val_loader,
                                decoder=decoder,
                                criterion=criterion)
        print(job_name)

        # Check if there was an improvement
        logging.info('Epoch: ' + str(epoch))
        if 'GB' in model_name:
            is_best = recent_f1 > best_f1
            best_f1 = max(recent_f1, best_f1)
            logging.info('Recent F1: ' + str(recent_f1))
            logging.info('Best F1: ' + str(best_f1))
        else:
            is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        logging.info('Recent BLEU4: '+str(recent_bleu4))
        logging.info('Best BLEU4: ' + str(best_bleu4))
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        scheduler.step()

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer, recent_bleu4, is_best)


def fine_tune(train_loader, decoder, criterion, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion: loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode 

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    metrics = Metrics()


    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, chexpert_labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        chexpert_labels = chexpert_labels.to(device)

        # Forward prop.
        scores, caps_sorted, decode_lengths, chexpert_scores, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets,  decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)
        loss = (1-chexpert_c) * loss + chexpert_c * criterion(chexpert_scores.view(-1, 3), chexpert_labels.view(-1))
        metrics.update(chexpert_scores.to('cpu'), chexpert_labels.to('cpu'))

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            print('Micro Positive F1 {f1:.4f}'.format(f1=metrics.calculate_metrics()['Micro Positive F1']))




def train(train_loader, decoder, criterion, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion: loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    metrics = Metrics()
    metrics_roc = MetricsROC()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, chexpert_labels) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        chexpert_labels = chexpert_labels.to(device)

        # Forward prop.
        if 'GB' in model_name:
            scores, chexpert_scores = decoder(imgs, caps, caplens)
        elif 'Multi' in model_name:
            scores, chexpert_scores = decoder(imgs, caps, caplens)
        else:
            scores = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps[:, 1:]
        decode_lengths = (caplens.squeeze(1) - 1).tolist()

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
        targets = pack_padded_sequence(targets,  decode_lengths, batch_first=True, enforce_sorted=False).data

        # Calculate loss
        loss = criterion(scores, targets)

        if 'GB' in model_name:
            loss += chexpert_c * criterion(chexpert_scores.view(-1, 3), chexpert_labels.view(-1))
            metrics.update(chexpert_scores.to('cpu'), chexpert_labels.to('cpu'))


        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses,
                                                                      top5=top5accs))
            if 'Multi' in model_name:
                print('Micro AUCROC {roc:.4f}'.format(roc=metrics.calculate_metrics()['Micro AUCROC']))
            if 'GB' in model_name:
                print('Micro Positive F1 {f1:.4f}'.format(f1=metrics.calculate_metrics()['Micro Positive F1']))

        # if i > 15:
        #     break
    logging.info('End of Epoch: [{0}][{1}/{2}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses,
                                                                    top5=top5accs))
    if 'GB' in model_name :
        logging.info('Micro Positive F1 {f1:.4f}'.format(f1=metrics.calculate_metrics()['Micro Positive F1']))
    if 'Multi' in model_name:
        logging.info('Micro AUCROC {roc:.4f}'.format(roc=metrics.calculate_metrics()['Micro AUCROC']))


def validate(val_loader, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    metrics = Metrics()
    metrics_roc = MetricsROC()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, chexpert_labels) in enumerate(val_loader):
            # print(caplens)

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            chexpert_labels = chexpert_labels.to(device)

            # Forward prop.
            if 'GB' in model_name:
                scores, chexpert_scores = decoder(imgs, caps, caplens)
            elif 'Multi' in model_name:
                scores, chexpert_scores = decoder(imgs, caps, caplens)
            else:
                scores = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps[:, 1:]
            decode_lengths = (caplens.squeeze(1) - 1).tolist()

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True, enforce_sorted=False).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False).data

            # Calculate loss
            loss = criterion(scores, targets)

            if 'GB' in model_name:
                loss += chexpert_c * criterion(chexpert_scores.view(-1, 3), chexpert_labels.view(-1))


            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = caps
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = [
                    [w for w in img_caps if w not in {word_map['**START**'], word_map['**PAD**']}]]  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds_list = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds_list):
                temp_preds.append(preds_list[j][:decode_lengths[j]])  # remove pads

            preds_list = temp_preds
            hypotheses.extend(preds_list)

            assert len(references) == len(hypotheses)

            if 'GB' in model_name:
                chexpert_scores = decoder.chexpert(preds, caplens)
                metrics.update(chexpert_scores.to('cpu'), chexpert_labels.to('cpu'))

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
                if 'GB' in model_name:
                    print('Micro Positive F1 {f1:.4f}'.format(f1=metrics.calculate_metrics()['Micro Positive F1']))

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        if 'GB' in model_name:
            logging.info('Micro Positive F1 {f1:.4f}'.format(f1=metrics.calculate_metrics()['Micro Positive F1']))

    if 'GB' in model_name:
        return bleu4, metrics.calculate_metrics()['Micro Positive F1']
    return bleu4



if __name__ == '__main__':
    main()