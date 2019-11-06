
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
import math
from torch.autograd import Variable  


# In[ ]:


class CnnModel(nn.Module):
    
    def __init__(self, args):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.kernel_size -- kernel size of the conv1d
        args.layer_num -- number of CNN layers
        """
        super(CnnModel, self).__init__()

        self.args = args
        if args.kernel_size % 2 == 0:
            raise ValueError("args.kernel_size should be an odd number")
            
        self.conv_layers = nn.Sequential()
        for i in range(args.layer_num):
            if i == 0:
                input_dim = args.embedding_dim
            else:
                input_dim = args.hidden_dim
            self.conv_layers.add_module('conv_layer{:d}'.format(i), nn.Conv1d(in_channels=input_dim, 
                                                  out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                             padding=(args.kernel_size-1)/2))
            self.conv_layers.add_module('relu{:d}'.format(i), nn.ReLU())
        
    def forward(self, embeddings):
        """
        Given input embeddings in shape of (batch_size, sequence_length, embedding_dim) generate a 
        sentence embedding tensor (batch_size, sequence_length, hidden_dim)
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)       
        """
        embeddings_ = embeddings.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
        hiddens = self.conv_layers(embeddings_)
        return hiddens


# In[ ]:


class RnnModel(nn.Module):

    def __init__(self, args, input_dim):
        """
        args.hidden_dim -- dimension of filters
        args.embedding_dim -- dimension of word embeddings
        args.layer_num -- number of RNN layers   
        args.cell_type -- type of RNN cells, GRU or LSTM
        """
        super(RnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == 'GRU':
            self.rnn_layer = nn.GRU(input_size=input_dim, 
                                    hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == 'LSTM':
            self.rnn_layer = nn.LSTM(input_size=input_dim, 
                                     hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
    def forward(self, embeddings, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
            mask -- a float tensor of masks, (batch_size, length)
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
        
        hidden, _ = self.rnn_layer(inputs_) #(sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden) #(length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0) #(batch_size, hidden_dim, sequence_length)


# In[ ]:




