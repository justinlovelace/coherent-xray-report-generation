import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import sys


class ReportDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'train', 'val', 'test'}

        self.data_folder = data_folder

        file_path = os.path.join(data_folder, 'df_master_{}.csv'.format(split))
        df_data = pd.read_csv(file_path)

        self.word2ind = np.load(os.path.join(data_folder, 'word2ind.npy'), allow_pickle=True).item()

        self.study_id =[]
        self.dicom_path = []
        self.dicom_features = []
        self.study_tokens = []
        self.study_text = []
        self.study_lens = []
        self.med_columns = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other', 'Fracture', 'Support Devices']
        self.medical_labels = []
        for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
            report_path = row['report_path']
            with open(os.path.join(data_folder, report_path), 'r') as f:
                report = f.read().split()
            self.dicom_features.append(row['processed_dicom_path'])
            self.medical_labels.append([row[x] if row[x] >= 0 else 2 for x in self.med_columns])
            self.study_id.append(row['study_id'])
            self.dicom_path.append(row['dicom_path'])
            self.study_tokens.append([self.word2ind['**START**']]+[self.word2ind[x] for x in report]+[self.word2ind['**END**']])
            self.study_text.append(report)
            self.study_lens.append(len(report)+2)

        self.pad_len = max(self.study_lens)

        # Total number of datapoints
        self.dataset_size = len(self.study_id)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        medical_labels = torch.LongTensor(self.medical_labels[i])

        dicom_ft = torch.load(os.path.join(self.data_folder, self.dicom_features[i])).permute(1, 2, 0)

        report = torch.LongTensor(self.study_tokens[i]+[self.word2ind['**PAD**']]*(self.pad_len-len(self.study_tokens[i])))

        study_len = torch.LongTensor([self.study_lens[i]])

        return dicom_ft, report, study_len, medical_labels

    def __len__(self):
        return self.dataset_size