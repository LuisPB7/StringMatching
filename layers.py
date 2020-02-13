import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import math
torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class PenalizedTanh(nn.Module):
    def forward(self, inp):
        a = torch.tanh(inp)
        return torch.where(inp>0, a, 0.25*a)
    
class HardAlignmentAttention(nn.Module):
    def __init__(self):
        super(HardAlignmentAttention, self).__init__()
        
    def forward(self, h1, h2):

        M = torch.bmm(h1, h2.permute((0,2,1)))
        W1 = F.softmax(M, dim=1)
        W2 = F.softmax(M, dim=2)
        W2 = W2.permute((0,2,1))
        in1_aligned = torch.bmm(W1.permute((0,2,1)) ,h1)
        in2_aligned = torch.bmm(W2.permute((0,2,1)), h2)
        s1_rep = torch.cat((h1, in2_aligned), dim=1)
        s2_rep = torch.cat((h2, in1_aligned), dim=1)

        s1_rep = nn.MaxPool2d( (s1_rep.size()[1] ,1) )(s1_rep).squeeze()
        s2_rep = nn.MaxPool2d( (s2_rep.size()[1] ,1) )(s2_rep).squeeze()
        return s1_rep.cuda(), s2_rep.cuda()

class GRUPenTanhCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUPenTanhCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3* hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size,3* hidden_size, bias=bias)
        self.reset_parameters()
        self.pen_tanh = PenalizedTanh()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = self.pen_tanh(i_r + h_r)
        inputgate = self.pen_tanh(i_i + h_i)
        newgate = self.pen_tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy
    
class GRUPenTanh(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRUPenTanh, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.forward_gru_cell = GRUPenTanhCell(input_dim, hidden_dim)
        self.backward_gru_cell = GRUPenTanhCell(input_dim, hidden_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        h0_forward = Variable(torch.zeros(1, x.size(0), self.hidden_dim).cuda())
        h0_backward = Variable(torch.zeros(1, x.size(0), self.hidden_dim).cuda())

        outs = []

        hn_forward = h0_forward[0,:,:]
        hn_backward = h0_backward[0,:,:]

        for seq in range(x.size(1)):
            hn_forward = self.forward_gru_cell(x[:,seq,:], hn_forward)
            hn_backward = self.backward_gru_cell(x[:,x.size(1)-seq-1,:], hn_backward)
            hn = torch.cat([hn_forward, hn_backward], dim=1)
            outs.append(hn)


        out = outs[-1].squeeze()
        outs = torch.stack(outs, dim=1)
        return outs, out
