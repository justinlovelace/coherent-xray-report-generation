import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from dataset import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
from nlgeval import NLGEval
import csv
from pathlib import Path

# Parameters
data_folder = '/home/ugrads/j/justinlovelace/MIMIC/cxr/data'  # folder with data files saved by create_input_files.py
torch.set_num_threads(4)
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
epochs = 64  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
decoder_lr = 5e-5  # learning rate for decoder
grad_clip = 1.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 250  # print training/validation stats every __ batches
checkpoint = None  # path to checkpoint, None if none
split = 'val'

nhead = 8
d_model = 256
dim_feedforward = 4096
num_encoder_layers = 1
num_decoder_layers = 6

chexpert_c = 0.9
temperature = 1
fine_tune_lr = 1e-5  # learning rate for decoder
beta = 1


if model_name == 'TX':
    job_name = "{}_bs{}_lr{}_nhead{}_dmodel{}_dimff{}_enclayers{}_declayers{}_clip{}".format(model_name, batch_size, decoder_lr, nhead,
                                                                             d_model, dim_feedforward,
                                                                             num_encoder_layers, num_decoder_layers, grad_clip)
elif 'TXGB' in model_name:
    job_name = "{}_bs{}_lr{}_finetunelr{}_nhead{}_dmodel{}_dimff{}_enclayers{}_declayers{}_chexpert{}_temp{}_beta{}".format(model_name, batch_size, decoder_lr, fine_tune_lr,
                                                                             nhead, d_model, dim_feedforward,
                                                                             num_encoder_layers, num_decoder_layers, chexpert_c, temperature, beta)

data_name = os.path.join(data_folder, 'saved_models', job_name)
set_logger(os.path.join(data_name, split + '.log'))

checkpoint = os.path.join(data_name, 'BEST_checkpoint.pth.tar')


# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()

# Load word map (word2ix)
word_map = np.load(os.path.join(data_folder, 'word2ind.npy'), allow_pickle=True).item()
ind2word = np.load(os.path.join(data_folder, 'ind2word.npy'), allow_pickle=True).item()

vocab_size = len(word_map)-2

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        ReportDataset(data_folder, split), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (imgs, caps, caplens, _) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['**START**']]] * k).to(device)  # (k, 1)


        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        with torch.no_grad():
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:

                temp_caplens = torch.full(size=(k,), fill_value=step, dtype=torch.long, device='cuda')
                # print(temp_caplens)
                # print(k_prev_words)
                imgs_batched = imgs.expand(k, 8, 8, 1024)
                if 'GB' in model_name:
                    scores = decoder(imgs_batched, k_prev_words, temp_caplens, eval=True)
                else:
                    scores = decoder(imgs_batched, k_prev_words, temp_caplens)
                scores = scores[:, step-1, :]
                scores = F.log_softmax(scores, dim=1)

                # Add
                scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

                # For the first step, all k points will have the same scores (since same k previous words, h, c)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words / vocab_size  # (s)
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                   next_word != word_map['**END**']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # Set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                k -= len(complete_inds)  # reduce beam length accordingly

                # Proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = seqs.clone().detach()

                # Break if things have been going on too long
                if step >= 400:
                    complete_seqs.extend(seqs[incomplete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[incomplete_inds])
                    break
                step += 1

            # Normalize scores by length
            for i in range(len(complete_seqs_scores)):
                seq_len = len(complete_seqs[i])
                # print('seq len {}, old score {}, new score {}'.format(seq_len, complete_seqs_scores[i], complete_seqs_scores[i]/seq_len))
                complete_seqs_scores[i] = complete_seqs_scores[i]/seq_len


            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]


            # References
            img_caps = caps[0].tolist()

            img_captions = [ind2word[w] for w in img_caps if w not in {word_map['**START**'], word_map['**END**'], word_map['**PAD**']}]  # remove <start> and pads
            references.append(' '.join(img_captions))

            # Hypotheses
            hypotheses.append(' '.join([ind2word[w] for w in seq if w not in {word_map['**START**'], word_map['**END**'], word_map['**PAD**']}]))

            assert len(references) == len(hypotheses)

    print('Saving generated reports')
    csv_file_path = os.path.join(data_name, split, 'reports.csv')
    Path(os.path.dirname(csv_file_path)).mkdir(parents=True, exist_ok=True)
    with open(csv_file_path, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for hyp in hypotheses:
            wr.writerow([hyp])

    # Calculate BLEU-4 scores
    print("Loading NLG model...")
    nlgeval = NLGEval(metrics_to_omit=['SkipThoughtCS', 'EmbeddingAverageCosineSimilarity', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])  # loads the models
    print("Computing NLG metrics...")
    metrics_dict = nlgeval.compute_metrics([references], hypotheses)
    print(metrics_dict)
    logging.info(str(metrics_dict))

    return metrics_dict


if __name__ == '__main__':
    beam_size = 4
    logging.info(job_name)
    logging.info(split)
    print("\nMetrics @ beam size of %d is %s" % (beam_size, str(evaluate(beam_size))))
