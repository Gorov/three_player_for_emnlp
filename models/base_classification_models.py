
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
# from models.models import CnnModel, RnnModel

# from basic_nlp_models import BasicNLPModel
# from models.encoder import Encoder, ClassificationEncoder
from models.rnn_model import RnnModel


# In[ ]:


class StackedCnnModel(nn.Module):
    
    def __init__(self, args):
        super(StackedCnnModel, self).__init__()

        self.args = args
        if args.kernel_size % 2 == 0:
            raise ValueError("args.kernel_size should be an odd number")
            
        self.conv_layer1 = nn.Sequential()
        input_dim = args.embedding_dim
        self.conv_layer1.add_module('conv_layer1', nn.Conv1d(in_channels=input_dim, 
                                              out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                         padding=(args.kernel_size-1)/2))
        self.conv_layer1.add_module('relu1', nn.ReLU())
        
        self.conv_layer2 = nn.Sequential()
        input_dim = args.hidden_dim
        self.conv_layer2.add_module('conv_layer2', nn.Conv1d(in_channels=input_dim, 
                                              out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                         padding=(args.kernel_size-1)/2))
        self.conv_layer2.add_module('relu2', nn.ReLU())
        
    def forward(self, embeddings, mask):
        """
        Given input embeddings in shape of (batch_size, sequence_length, embedding_dim) generate a 
        sentence embedding tensor (batch_size, sequence_length, hidden_dim)
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim)
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length)       
        """
        masked_input = embeddings * mask.unsqueeze(-1) #(batch_size, sequence_length, embedding_dim)        
        embeddings_ = masked_input.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
        hiddens1 = self.conv_layer1(embeddings_)
        hiddens2 = self.conv_layer2(hiddens1)
        return hiddens1, hiddens2


# In[ ]:


class BasicClassificationModel(nn.Module):
    
    def __init__(self, embeddings, args):
        super(BasicClassificationModel, self).__init__()
        self.args = args
        self.pos_embedding_dim = args.pos_embedding_dim
        self.model_type = args.model_type
                    
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)
        
        if args.pos_embedding_dim > 0:
            self.pos_embed_layer = self._create_pos_embed_layer()
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        if args.pos_embedding_dim > 0:
            input_dim = args.embedding_dim + args.pos_embedding_dim
        else:
            input_dim = args.embedding_dim
            
        if self.model_type == 'CNN':
            self.conv_layer = nn.Sequential()
            self.conv_layer.add_module('conv_layer', nn.Conv1d(in_channels=input_dim, 
                                                  out_channels=args.hidden_dim, kernel_size=args.kernel_size,
                                                                             padding=(args.kernel_size-1)/2))
            self.conv_layer.add_module('relu', nn.ReLU())
        else:
            self.conv_layer = RnnModel(args, input_dim)
        
#         self.dropout = nn.Dropout(args.dropout_rate)
        
#         self.pred_layer = nn.Sequential()
#         self.pred_layer.add_module('linear1', nn.Linear(self.hidden_dim, self.mlp_hidden_dim))
#         self.pred_layer.add_module('tanh', nn.Tanh())
#         self.pred_layer.add_module('linear2', nn.Linear(self.mlp_hidden_dim, self.num_labels))
        
        self.pred_layer = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.loss_func = nn.CrossEntropyLoss()

    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = self.args.fine_tuning
        return embed_layer
    
    
    def _create_pos_embed_layer(self):
        embed_layer = nn.Embedding(3, self.pos_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
            
        
    def forward(self, x, e, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            e -- postions (batch_size, length) , currently {0, 1, 2}
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """
        embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        if self.pos_embedding_dim > 0:
            pos_embeddings = self.pos_embed_layer(e) #(batch_size, length, 3)
            embeddings = torch.cat([embeddings, pos_embeddings], dim=2)
        
        if self.model_type == 'CNN':
            masked_input = embeddings * mask.unsqueeze(-1)
            masked_input = masked_input.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
            hiddens = self.conv_layer(masked_input) #(batch_size, hidden_dim, sequence_length) 
        else:
            hiddens = self.conv_layer(embeddings, mask)
        
#         hiddens = self.dropout(hiddens)
        
#         avg_hidden_ = torch.mean(hiddens, dim=2)
#         avg_hidden_drop = self.dropout(avg_hidden_)
        neg_inf = -1.0e6
        max_hidden = torch.max(hiddens + (1 - mask).unsqueeze(1) * neg_inf, dim=2)[0]
        
        predict = self.pred_layer(max_hidden)

        return predict


# In[ ]:


class BasicRankingModel(BasicClassificationModel):
    
    def __init__(self, embeddings, args):
        super(BasicRankingModel, self).__init__(embeddings, args)
        self.gamma = 2.0
        self.m_positive = 2.5
        self.m_negative = 0.5
        
        print 'use cuda:', self.args.cuda
        
    def _set_none_relation_id(self, none_id):
        self.none_relation_id = none_id
        self.none_relation_id = Variable(torch.from_numpy(np.array([self.none_relation_id])))
        if self.args.cuda:
            self.none_relation_id = self.none_relation_id.cuda()
            
    def loss_func(self, scores, y):
        y_matrix = torch.zeros_like(scores)
        y_matrix.scatter_(dim=1, index=y.unsqueeze(1), value=1.0)
        
        pos_scores = torch.sum(scores * y_matrix, dim=1) # (batch,)
        
        neg_inf = -1.0e6
        neg_scores = scores * (1-y_matrix) + neg_inf * y_matrix
        max_neg_scores = torch.max(neg_scores, dim=-1)[0] # (batch,)
        
        pred_loss = torch.log(1 + torch.exp(self.gamma * (self.m_positive - pos_scores)))
        pred_loss += torch.log(1 + torch.exp(self.gamma * (self.m_negative + max_neg_scores)))

        return torch.mean(pred_loss)
    
    def loss_func_no_other(self, scores, y):
        y_matrix = torch.zeros_like(scores)
        y_matrix.scatter_(dim=1, index=y.unsqueeze(1), value=1.0)
        
        none_matrix = torch.zeros_like(scores)
#         print self.none_relation_id
        none_idx = self.none_relation_id.expand(scores.size(0))
#         print none_idx
        none_matrix.scatter_(dim=1, index=none_idx.unsqueeze(1), value=1.0)
        
        pos_scores = torch.sum(scores * y_matrix, dim=1) # (batch,)
        
        neg_inf = -1.0e6
        neg_scores = scores * (1-y_matrix) + neg_inf * y_matrix
        neg_scores = neg_scores * (1-none_matrix) + neg_inf * none_matrix
        max_neg_scores = torch.max(neg_scores, dim=-1)[0] # (batch,)
        
        none_as_gold_mask = (y == self.none_relation_id).type(torch.FloatTensor)
        if self.args.cuda:
            none_as_gold_mask = none_as_gold_mask.cuda()
        
        pred_loss = torch.log(1 + torch.exp(self.gamma * (self.m_positive - pos_scores))) * (1 - none_as_gold_mask)
        pred_loss += torch.log(1 + torch.exp(self.gamma * (self.m_negative + max_neg_scores)))

        return torch.mean(pred_loss)
    
    def predict_no_other(self, scores):
        
        none_matrix = torch.zeros_like(scores)
        none_idx = self.none_relation_id.expand(scores.size(0))
        none_matrix.scatter_(dim=1, index=none_idx.unsqueeze(1), value=1.0)
        
        neg_inf = -1.0e6
        masked_scores = scores * (1-none_matrix) + neg_inf * none_matrix
        max_scores, y_pred = torch.max(masked_scores, dim=-1) # (batch,)
        max_scores = max_scores.cpu().data.numpy()
        y_pred = y_pred.cpu().data.numpy()
        
        none_relation_id = self.none_relation_id.cpu().data.numpy()[0]
        
        for i in range(len(y_pred)):
            if max_scores[i] < 0:
                y_pred[i] = none_relation_id
                max_scores[i] = 0.0
        
        return max_scores, y_pred
        
        


# In[ ]:


class SingleHeadAttentionModel(BasicClassificationModel):
    
    def __init__(self, embeddings, args):
        super(SingleHeadAttentionModel, self).__init__(embeddings, args)
        
#         self.attention_layer = nn.Sequential()
#         self.attention_layer.add_module('linear1', nn.Linear(self.hidden_dim, self.mlp_hidden_dim))
#         self.attention_layer.add_module('tanh', nn.Tanh())
#         self.attention_layer.add_module('linear2', nn.Linear(self.mlp_hidden_dim, 1))
        
        self.attention_layer = nn.Linear(self.hidden_dim, 1)

        
    def forward(self, x, e, mask):
        """
        Inputs:
            q -- torch Variable in shape of (batch_size, max length of the sequence in the batch)
            p -- torch Variable in shape of (batch_size*(1 + num_negative_sample),
                                                max length of the sequence in the batch)
        Outputs:
            predict -- (batch_size, (1 + num_negative_sample)) classification prediction
        """
        embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        if self.pos_embedding_dim > 0:
            pos_embeddings = self.pos_embed_layer(e) #(batch_size, length, 3)
            embeddings = torch.cat([embeddings, pos_embeddings], dim=2)
        
        if self.model_type == 'CNN':
            masked_input = embeddings * mask.unsqueeze(-1)
            masked_input = masked_input.transpose(1, 2) #(batch_size, embedding_dim, sequence_length)
            hiddens = self.conv_layer(masked_input) #(batch_size, hidden_dim, sequence_length) 
        else:
            hiddens = self.conv_layer(embeddings, mask)
        
        attention_scores_ = self.attention_layer(hiddens.transpose(1, 2)) #(batch_size, sequence_length, 1)
        
        neg_inf = -1.0e6
        attention_scores_ = attention_scores_ + (1 - mask).unsqueeze(2) * neg_inf
        
        attention_scores = attention_scores_.transpose(1, 2) #(batch_size, 1, sequence_length) 
        
        attention_softmax = F.softmax(attention_scores, dim=2)
        
        avg_hidden_ = torch.mean(hiddens * attention_softmax, dim=2)
#         max_hidden = torch.max(hiddens * attention_softmax + (1 - mask).unsqueeze(1) * neg_inf, dim=2)[0]
        
        predict = self.pred_layer(avg_hidden_)
#         predict = self.pred_layer(max_hidden)

        return predict


# In[ ]:


class BasicAttentionModel(nn.Module):
    
    def __init__(self, embeddings, args):
        super(BasicAttentionModel, self).__init__()
        self.args = args
                    
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.encoder = StackedCnnModel(args)
        
        self.attention_layer = nn.Sequential()
        self.attention_layer.add_module('linear1', nn.Linear(self.hidden_dim, self.mlp_hidden_dim))
        self.attention_layer.add_module('tanh', nn.Tanh())
        self.attention_layer.add_module('linear2', nn.Linear(self.mlp_hidden_dim, 1))
        
        self.pred_layer = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.loss_func = nn.CrossEntropyLoss()

    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = self.args.fine_tuning
        return embed_layer
            
    def forward(self, x, mask):
        """
        Inputs:
            q -- torch Variable in shape of (batch_size, max length of the sequence in the batch)
            p -- torch Variable in shape of (batch_size*(1 + num_negative_sample),
                                                max length of the sequence in the batch)
        Outputs:
            predict -- (batch_size, (1 + num_negative_sample)) classification prediction
        """
        embed = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        # get rationales
        hiddens1, hiddens2 = self.encoder(embed, mask) #(batch_size, hidden_dim, sequence_length) 
        
        hiddens = torch.cat([hiddens1, hiddens2], dim=2) #(batch_size, hidden_dim, sequence_length * 2) 
        
        attention_scores_ = self.attention_layer(hiddens.transpose(1, 2)) #(batch_size, sequence_length * 2, 1)
        attention_scores = attention_scores_.transpose(1, 2) #(batch_size, 1, sequence_length * 2)
        attention_softmax = F.softmax(attention_scores, dim=2)
        
        avg_hidden_ = torch.mean(hiddens * attention_softmax, 2)
        
        predict = self.pred_layer(avg_hidden_)

        return predict


# In[ ]:


class IntraAttentionModel(BasicAttentionModel):
    def __init__(self, embeddings, args):
        super(IntraAttentionModel, self).__init__(embeddings, args)
        
        self.attention_layer = nn.Sequential()
        self.attention_layer.add_module('linear1', nn.Linear(self.hidden_dim * 2, self.mlp_hidden_dim))
        self.attention_layer.add_module('tanh', nn.Tanh())
        self.attention_layer.add_module('linear2', nn.Linear(self.mlp_hidden_dim, 1))
        
        self.pred_layer = nn.Linear(self.hidden_dim * 2, self.num_labels)
        
        
    def _get_intra_attention_matrix(self, hiddens):
        """
        This function takes the sequences of hidden states and return 
        the word-by-word attention score matrix. 
        
        Inputs:
            hiddens -- (batch_size, hidden_dim, sequence_length * 2) 
                                                
        Outputs:
            intra_matching_states -- (batch_size, sequence_length * 2, hidden_dim * 2)
        """
        
        attention_matrix = torch.bmm(hiddens.transpose(1, 2), hiddens)
        mask = Variable(torch.eye(hiddens.size(2)))
        if self.args.cuda:
            mask = mask.cuda()
        neg_inf = -1.0e6
        
#         attention_softmax = F.softmax(attention_matrix, dim=2)
        attention_softmax = F.softmax(attention_matrix + mask.unsqueeze(0) * neg_inf,
                                     dim=2)
        
        # shape: (batch_size, sequence_length * 2, hidden_dim)
        hiddens_tilda = torch.bmm(attention_softmax, hiddens.transpose(1, 2))
                
        # shape: (batch_size, sequence_length * 2, hidden_dim * 2)
        intra_matching_states = torch.cat([hiddens.transpose(1, 2).contiguous(), hiddens_tilda], dim=2)
        return intra_matching_states        
        
        
    def forward(self, x, mask):
        """
        Inputs:
            q -- torch Variable in shape of (batch_size, max length of the sequence in the batch)
            p -- torch Variable in shape of (batch_size*(1 + num_negative_sample),
                                                max length of the sequence in the batch)
        Outputs:
            predict -- (batch_size, (1 + num_negative_sample)) classification prediction
        """
        embed = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        # get rationales
        hiddens1, hiddens2 = self.encoder(embed, mask) #(batch_size, hidden_dim, sequence_length) 
        
        hiddens = torch.cat([hiddens1, hiddens2], dim=2) #(batch_size, hidden_dim, sequence_length * 2) 
        
        intra_matching_states = self._get_intra_attention_matrix(hiddens)
        
        attention_scores = self.attention_layer(intra_matching_states) #(batch_size, sequence_length * 2, 1)
        attention_softmax = F.softmax(attention_scores, dim=2)
        
        avg_hidden_ = torch.mean(intra_matching_states * attention_softmax, dim=1)
        
        predict = self.pred_layer(avg_hidden_)

        return predict

