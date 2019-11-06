
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.rnn_model import CnnModel, RnnModel


# In[ ]:


def _get_entropy(p):
    """
    Compute entropy of the input prob vector p
    Inputs:
        p -- torch variable, a list of prob vectors
    Outputs:
        entropy -- the average entropy of all the vectors
    """
    log_p = torch.log(p)
    return torch.mean(- p * log_p)

def single_soft_regularization_loss(z):
    """
    Compute regularization loss, based on a given rationale sequence
    Inputs:
        z -- torch variable, "binary" rationale, (batch_size, sequence_length)
    Outputs:
        a loss value that contains two parts:
        continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
        sparsity_loss -- total number of ones
    """
    z_ = torch.cat([z[:, 1:], z[:, -1:]], dim=-1)
    continuity_loss = torch.mean(torch.mean(torch.abs(z - z_), dim=-1))
    sparsity_loss = _get_entropy(z)

    return continuity_loss, sparsity_loss


# In[ ]:


class Generator(nn.Module):
    
    def __init__(self, args, input_dim):
        """        
        args.z_dim -- rationale or not, always 2
        args.model_type -- "CNN" or "RNN"

        if CNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.kernel_size -- kernel size of the conv1d
            args.layer_num -- number of CNN layers        
        if use RNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.layer_num -- number of RNN layers   
            args.cell_type -- type of RNN cells, "GRU" or "LSTM"
        """
        super(Generator, self).__init__()
        
        self.args = args
        self.z_dim = args.z_dim
        
        if args.model_type == "CNN":
            self.generator_model = CnnModel(args, input_dim)
        elif args.model_type == "RNN":
            self.generator_model = RnnModel(args, input_dim)
        self.output_layer = nn.Linear(args.hidden_dim, self.z_dim)
        
        
    def forward(self, x, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.generator_model(x, mask).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)

        return scores
    


# In[ ]:


class SoftGenerator(nn.Module):
    
    def __init__(self, args):
        """        
        args.z_dim -- rationale or not, always 2
        args.gumbel_temprature -- temprature
        args.model_type -- "CNN" or "RNN"

        if CNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.kernel_size -- kernel size of the conv1d
            args.layer_num -- number of CNN layers        
        if use RNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.layer_num -- number of RNN layers   
            args.cell_type -- type of RNN cells, "GRU" or "LSTM"
        """
        super(SoftGenerator, self).__init__()
        
        self.args = args
        
        if args.model_type == "CNN":
            self.generator_model = CnnModel(args)
        elif args.model_type == "RNN":
            self.generator_model = RnnModel(args)
        self.output_layer = nn.Linear(args.hidden_dim, 2)
        
        
    def forward(self, x):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output soft attention weights, (batch_size, sequence_length)
        """
        hiddens = self.generator_model(x).transpose(1, 2).contiguous() #(batch_size, sequence_length, hidden_dim)
        
        scores = self.output_layer(hiddens)#(batch_size, sequence_length, 2)
        probs = F.softmax(scores, dim=-1)
        z = probs[:, :, 1]
        
        return z
    


# In[ ]:


class BernoulliSoftGenerator(nn.Module):
    pass


# In[ ]:


class DepRnnModel(nn.Module):

    def __init__(self, args, input_dim):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.layer_num -- number of RNN layers   
        args.cell_type -- type of RNN cells, GRU or LSTM
        """
        super(DepRnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=input_dim, 
                                    hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=input_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
    def forward(self, embeddings, h0=None, c0=None, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
            h0, c0 --  (num_layers * num_directions, batch, hidden_size)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)
        """
        embeddings_ = embeddings.transpose(0, 1) #(sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = map(int, seq_lengths)
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        if self.args.cell_type == 'GRU' and h0 is not None:
            hidden, _ = self.rnn_layer(inputs_, h0)
        elif self.args.cell_type == 'LSTM' and h0 is not None and c0 is not None:
            hidden, _ = self.rnn_layer(inputs_, (h0, c0)) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        else:
            hidden, _ = self.rnn_layer(inputs_)
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)


# In[ ]:


class DepGenerator(nn.Module):
    
    def __init__(self, args, input_dim):
        """        
        args.z_dim -- rationale or not, always 2
        args.model_type -- "CNN" or "RNN"

        if CNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.kernel_size -- kernel size of the conv1d
            args.layer_num -- number of CNN layers        
        if use RNN:
            args.hidden_dim -- dimension of filters
            args.embedding_dim -- dimension of word embeddings
            args.layer_num -- number of RNN layers   
            args.cell_type -- type of RNN cells, "GRU" or "LSTM"
        """
        super(DepGenerator, self).__init__()
        
        self.args = args
        self.z_dim = args.z_dim
        
        self.rnn_model = DepRnnModel(args, input_dim)
        self.output_layer = nn.Linear(args.hidden_dim, self.z_dim)
        
        
    def forward(self, x, h0=None, c0=None, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length)
        """
        
        #(batch_size, sequence_length, hidden_dim)
        hiddens = self.rnn_model(x, h0, c0, mask).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)

        return scores

