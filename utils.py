import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import csv
import numpy
import pickle
import copy

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Load character list #
with open("characters.pkl", 'rb') as char:
    characters = pickle.load(char)

class PadSequence:
    def __call__(self, batch):
        sequences_s1 = [x[0] for x in batch]
        sequences_s1_padded = torch.nn.utils.rnn.pad_sequence(sequences_s1, batch_first=True)
        sequences_s2 = [x[1] for x in batch]
        sequences_s2_padded = torch.nn.utils.rnn.pad_sequence(sequences_s2, batch_first=True)        
        lengths_s1 = torch.LongTensor([len(x) for x in sequences_s1])
        lengths_s2 = torch.LongTensor([len(x) for x in sequences_s2])
        labels = torch.FloatTensor([x[2] for x in batch])
        return sequences_s1_padded, sequences_s2_padded, lengths_s1, lengths_s2, labels

class StringMatchingDataset(Dataset):

    def __init__(self, name):
        super(StringMatchingDataset, self).__init__()
        self.data = []
        train_datasets = name.split(',')
        for dataset in train_datasets:
            self.data += list(csv.DictReader(open('datasets/{}.csv'.format(dataset)), delimiter='|', fieldnames=['s1', 's2', 'res']))
        self.characters = characters
        self.n_chars = len(characters)

    def stringToCharSeq(self, string):
        # Replace invalid characters
        string = string.replace("{", " ").replace("}", " ").replace("\\", " ").replace("$", " ")
        barray = list(bytearray(unicodedata.normalize('NFKD', string), encoding='utf-8'))
        tensor = torch.zeros(len(barray), self.n_chars)
        for i, ch in enumerate(barray): 
         try: tensor[i][self.characters.index(ch)] = 1.0
         except: print("ERROR WHEN PROCESSING CHARACTER " + repr(ch) + " IN " + string)
        return tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        res = int(self.data[index]['res'])
        s1 = self.stringToCharSeq(self.data[index]['s1'])
        s2 = self.stringToCharSeq(self.data[index]['s2'])
        return s1, s2, res
