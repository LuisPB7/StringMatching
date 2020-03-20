import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import math
torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

class MogrifierLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.mogrify_steps = mogrify_steps
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)        
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0: self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else: self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r        
        self.tanh = nn.Tanh()
        self.init_parameters()
        
    def init_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-std, std)
            
    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else: x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        """
        inp shape: (batch_size, input_size)
        each of states shape: (batch_size, hidden_size)
        """
        ht, ct = states
        x, ht = self.mogrify(x,ht)   # Note: This should be called every timestep
        gates = self.x2h(x) + self.h2h(ht)  # (batch_size, 4 * hidden_size)
        in_gate, forget_gate, new_memory, out_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        forget_gate = torch.sigmoid(forget_gate)
        out_gate = torch.sigmoid(out_gate)
        new_memory = self.tanh(new_memory)
        c_new = (forget_gate * ct) + (in_gate * new_memory)
        h_new = out_gate * self.tanh(c_new)
        return h_new, c_new


class MogrifierLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, steps=3, bias=True):
        super(MogrifierLSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.forward_cell = MogrifierLSTMCell(input_dim, hidden_dim, steps)
        self.backward_cell = MogrifierLSTMCell(input_dim, hidden_dim, steps)

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
        cn_forward = h0_forward[0,:,:]
        cn_backward = h0_backward[0,:,:]        
        for seq in range(x.size(1)):
            hn_forward, cn_forward = self.forward_cell(x[:,seq,:], (hn_forward,cn_forward))
            hn_backward, cn_backward = self.backward_cell(x[:,x.size(1)-seq-1,:], (hn_backward,cn_backward))
            hn = torch.cat([hn_forward, hn_backward], dim=1)
            outs.append(hn)
        out = outs[-1].squeeze()
        outs = torch.stack(outs, dim=1)
        return outs, out

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