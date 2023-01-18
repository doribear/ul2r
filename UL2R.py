from tokenizers import Tokenizer
from torch.utils.data import Dataset

import numpy as np
import torch
import pandas as pd

class UL2R_Denoiser():
    def __init__(self, tokenizer : str = 'tokenizer.json', noising_ratio : dict = {'R_noising' : 0.5, 'S_noising' : 0.25, 'X_noising' : 0.25}, mask_token : str = '[MASK]') -> None:
        
        self.tokenizer = Tokenizer.from_file(tokenizer)
        self.noising_ratio = noising_ratio
        assert self.noising_ratio['R_noising'] + self.noising_ratio['S_noising'] + self.noising_ratio['X_noising'] == 1, 'sum of noiging ratio must be equal "1"'
        self.noising_ratio_list = self.get_ratio_list()
        self.mask_ids = self.tokenizer.token_to_id(mask_token)
        self.pad_ids = self.tokenizer.token_to_id('[PAD]')

    def get_ratio_list(self):
        ratio_list = [self.noising_ratio['R_noising'], self.noising_ratio['S_noising'], self.noising_ratio['X_noising']]
        for i in range(1, 3):
            ratio_list[i] = ratio_list[i-1] + ratio_list[i]
        return ratio_list

    def r_noise(self, ids):
        mask = list(map(lambda x : self.mask_ids if np.random.rand() <= 0.15 and x != self.pad_ids else x, ids))
        return mask

    def s_noise(self, ids):
        sequence_len = len(list(filter(lambda x : x != self.pad_ids, ids)))
        mask_len = sequence_len // 2
        mask = list(map(lambda x : x, ids))
        if np.random.rand() > 0.5:
            mask[:mask_len] = [self.mask_ids for _ in range(mask_len)]
        else:
            mask[mask_len:sequence_len] = [self.mask_ids for _ in range(mask_len + 1)]
        mask = mask[:len(ids)]
        if len(mask) != len(ids):
            print('here')
        return mask

    def x_noise(self, ids):
        sequence_len = len(list(filter(lambda x : x != self.pad_ids, ids)))
        mask_len = sequence_len // 4
        mask = list(map(lambda x : x, ids))
        mask[np.random.randint(0, sequence_len - mask_len) : mask_len] = [self.mask_ids for _ in range(mask_len)]
        mask = list(map(lambda x : self.mask_ids if np.random.rand() <= 0.3 and x not in [self.pad_ids, self.mask_ids] else x, mask))
        mask = mask[:len(ids)]
        if len(mask) != len(ids):
            print('here')
        return mask

    def masking(self, ids, mlm):
        if mlm:
            mask = self.r_noise(ids)
        else:
            probe = np.random.rand()
            if probe <= self.noising_ratio_list[0]:
                mask = self.r_noise(ids)
            elif self.noising_ratio_list[0] < probe and probe <= self.noising_ratio_list[1]:
                mask = self.s_noise(ids)
            else:
                mask = self.x_noise(ids)
        return mask

    def __call__(self, text, mlm : bool = False):
        text = self.tokenizer.encode(text)
        ids, type_ids, attention_mask = text.ids, text.type_ids, text.attention_mask
        mask = self.masking(ids, mlm)
        return ids, mask, type_ids, attention_mask

class UL2R_dataset(Dataset):
    def __init__(self, corpus):
        super(UL2R_dataset, self).__init__()

        self.noiser = UL2R_Denoiser()
        self.corpus = corpus

    @classmethod
    def fromfile(cls, files : list = ['corpus/15_감성대화.txt', 'corpus/16_한국어 대화 요약.txt']):
        corpus = list(map(lambda x : list(map(lambda y : y.strip(), open(x, 'r').readlines())), files))
        corpus = sum(corpus, [])
        return cls(corpus)
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        ids, mask, type_ids, attention_mask = self.noiser(self.corpus[index])
        ids, mask, type_ids, attention_mask = torch.LongTensor(ids), torch.LongTensor(mask), torch.LongTensor(type_ids), torch.LongTensor(attention_mask)
        return ids, mask, type_ids, attention_mask

class MLM_dataset(Dataset):
    def __init__(self, corpus):
        super(MLM_dataset, self).__init__()

        self.noiser = UL2R_Denoiser()
        self.corpus = corpus

    @classmethod
    def fromfile(cls, files : list = ['corpus/15_감성대화.txt', 'corpus/16_한국어 대화 요약.txt']):
        corpus = list(map(lambda x : list(map(lambda y : y.strip(), open(x, 'r').readlines())), files))
        corpus = sum(corpus, [])
        return cls(corpus)
        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        ids, mask, type_ids, attention_mask = self.noiser(self.corpus[index], True)
        ids, mask, type_ids, attention_mask = torch.LongTensor(ids), torch.LongTensor(mask), torch.LongTensor(type_ids), torch.LongTensor(attention_mask)
        return ids, mask, type_ids, attention_mask


class Classify_dataset(Dataset):
    def __init__(self, tokenizer, corpus, label):
        super(Classify_dataset, self).__init__()

        self.tokenizer = tokenizer
        self.corpus = corpus
        self.label = label
        self.label_dict = Classify_dataset.get_label_dict(label)

    @staticmethod
    def get_label_dict(label):
        label_dict = {}
        labels = set(label)
        for i, k in enumerate(labels):
            label_dict[k] = i
        return label_dict

    @classmethod
    def fromfile(cls, tokenizer : str = 'tokenizer.json', file : str = '감성대화말뭉치.csv'):
        tokenizer = Tokenizer.from_file(tokenizer)
        data = pd.read_csv(file)
        corpus, label = data['sentence'].to_list(), data['label'].to_list()
        return cls(tokenizer, corpus, label)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        encoded = self.tokenizer.encode(self.corpus[index])
        ids, type_ids, attention_mask = torch.LongTensor(encoded.ids), torch.LongTensor(encoded.type_ids), torch.LongTensor(encoded.attention_mask)
        label = torch.LongTensor([self.label_dict[self.label[index]]])
        return ids, type_ids, attention_mask, label

