import torch
from torch import nn
from torchvision import transforms, datasets
from torch.autograd import Variable
from layers import PenalizedTanh, HardAlignmentAttention
from utils import StringMatchingDataset, PadSequence
from variables import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from collections import OrderedDict

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.dropout = nn.Dropout(DROPOUT_P)
        self.gru1 = nn.GRU(CHAR_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.gru2 = nn.GRU(2*HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True, bidirectional=True)
    
    def init_hidden(self, batch_size):
        directions = 2
        return Variable(torch.zeros((directions, batch_size, self.hidden_size))).cuda()
    
    def forward(self, s, lengths):
        # Zero initial hidden state for both GRUs
        hidden_1 = self.init_hidden(s.size()[0])
        hidden_2 = self.init_hidden(s.size()[0])

        # Dynamic sequence length
        s = torch.nn.utils.rnn.pack_padded_sequence(s, lengths, batch_first=True, enforce_sorted=False)

        s_rep, hidden_1 = self.gru1(s, hidden_1)
        s_rep, _ = torch.nn.utils.rnn.pad_packed_sequence(s_rep, batch_first=True)
        s_rep = self.dropout(s_rep)

        s_rep = torch.nn.utils.rnn.pack_padded_sequence(s_rep, lengths, batch_first=True, enforce_sorted=False)
        s_rep, _ = self.gru2(s_rep, hidden_2) # s_rep contains last hidden_state
        s_rep, _ = torch.nn.utils.rnn.pad_packed_sequence(s_rep, batch_first=True)
        s_rep = self.dropout(s_rep)
        return s_rep

class RSHardAttentionModel(pl.LightningModule):
    def __init__(self, name, train_dataset, test_dataset):
        super(RSHardAttentionModel, self).__init__()
        self.lin1 = nn.Linear(HIDDEN_SIZE*8, HIDDEN_SIZE)
        self.lin2 = nn.Linear(HIDDEN_SIZE, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_P)
        self.sigmoid = nn.Sigmoid()
        self.Encoder = Encoder()
        self.train_dataset =  train_dataset
        self.test_dataset = test_dataset
        self.name = name
        self.hard_attention = HardAlignmentAttention()
    
    def forward(self, s1, s2, s1_lens, s2_lens):
        # Representation for each input sentence
        s1_rep = self.Encoder(s1, s1_lens)
        s2_rep = self.Encoder(s2, s2_lens)

        # Hard Alignment Attention
        s1_rep, s2_rep = self.hard_attention(s1_rep, s2_rep)

        # Concatenate, multiply, subtract
        conc = torch.cat([s1_rep, s2_rep], 1)
        mul = s1_rep * s2_rep
        dif = torch.abs(s1_rep - s2_rep)
        final = torch.cat([conc, mul, dif], 1)
        final = self.dropout(final)

        # Linear layers and softmax
        final = self.lin1(final)
        final = self.relu(final)
        final = self.dropout(final)
        final = self.lin2(final)
        out = self.sigmoid(final)
        return out
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        outputs = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()).squeeze(1)
        labels = batch[4]
        loss = F.binary_cross_entropy(outputs, labels)
        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(StringMatchingDataset(self.train_dataset), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, collate_fn=PadSequence())
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(StringMatchingDataset(self.test_dataset), shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, collate_fn=PadSequence())
    
    def test_step(self, batch, batch_nb):
    
        # implement your own
        out = self.forward(batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()).squeeze(1)
        y = batch[4]
        loss_test = F.binary_cross_entropy(out, y)
    
        # calculate acc
        labels_hat = torch.tensor([round(o.item()) for o in out])
        
        num_true = 0.0
        num_false = 0.0
        num_true_predicted_true = 0.0
        num_true_predicted_false = 0.0
        num_false_predicted_true = 0.0
        num_false_predicted_false = 0.0
        for i in range(len(y)):
            if y[i] == 0:
                num_false += 1
                if labels_hat[i] == 0:
                    num_false_predicted_false += 1
                else:
                    num_false_predicted_true += 1
            else:
                num_true += 1
                if labels_hat[i] == 1:
                    num_true_predicted_true += 1
                else:
                    num_true_predicted_false += 1
    
        test_acc = (num_true_predicted_true + num_false_predicted_false) / (num_true + num_false)
        try:
            test_pre = (num_true_predicted_true) / (num_true_predicted_true + num_false_predicted_true)
        except:
            test_pre = 0
        try:
            test_rec = (num_true_predicted_true) / (num_true_predicted_true + num_true_predicted_false)
        except:
            test_rec = 0
        try:
            test_f1 = 2.0 * ((test_pre * test_rec) / (test_pre + test_rec))
        except:
            test_f1 = 0        
            
        # all optional...
        # return whatever you need for the collation function test_end
        output = OrderedDict({
            'test_loss': loss_test,
            'test_acc': torch.tensor(test_acc), # everything must be a tensor
            'test_prec': torch.tensor(test_pre),
            'test_rec': torch.tensor(test_rec),
            'test_f1': torch.tensor(test_f1),
        })
    
        # return an optional dict
        return output
    
    def test_end(self, outputs):
        """
        Called at the end of test to aggregate outputs
        :param outputs: list of individual outputs of each test step
        :return:
        """
        
        torch.save(self.state_dict(), 'weights/{}-{}.pt'.format(self.name, self.train_dataset))
        
        test_loss_mean = 0
        test_acc_mean = 0
        test_prec_mean = 0
        test_rec_mean = 0
        test_f1_mean = 0
        for output in outputs:
            test_loss_mean += output['test_loss']
            test_acc_mean += output['test_acc']
            test_prec_mean += output['test_prec']
            test_rec_mean += output['test_rec']
            test_f1_mean += output['test_f1']
    
        test_loss_mean /= len(outputs)
        test_acc_mean /= len(outputs)
        test_prec_mean /= len(outputs)
        test_rec_mean /= len(outputs)
        test_f1_mean /= len(outputs)
        tqdm_dict = {'test_loss': test_loss_mean.item(), 'test_acc': test_acc_mean.item()}
        print("Testing accuracy is {}".format(test_acc_mean.item()))
    
        # show test_loss and test_acc in progress bar but only log test_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'test_loss': test_loss_mean.item()}
        }
        
        with open("results.txt", "a") as myfile:
            myfile.write("MODEL NAME: {}\n".format(self.name))
            myfile.write("TRAINING DATASET: {}\n".format(self.train_dataset))
            myfile.write("TESTING DATASET: {}\n".format(self.test_dataset))
            myfile.write("Accuracy: {}\n".format(test_acc_mean))
            myfile.write("Precision: {}\n".format(test_prec_mean))
            myfile.write("Recall: {}\n".format(test_rec_mean))
            myfile.write("F1: {}\n".format(test_f1_mean))
            myfile.write("\n")
        
        return results    
