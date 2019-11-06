
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
from models.generator import Generator, DepGenerator

from utils.utils import single_regularization_loss_batch, bao_regularization_hinge_loss_batch
from utils.utils import bao_regularization_loss_batch, count_regularization_loss_batch
from utils.utils import count_regularization_hinge_loss_batch
from utils.utils import bao_regularization_hinge_loss_batch_with_none_loss

from rationale_3players_sentence_classification_models import HardIntrospection3PlayerClassificationModel

from collections import deque


# In[ ]:


def create_relative_pos_embed_layer(max_pos_num, pos_embedding_dim):
    embed_layer = nn.Embedding(2 * max_pos_num + 1, pos_embedding_dim)
    
    embed_layer.weight.data.normal_(mean=0, std=0.1)
    embed_layer.weight.requires_grad = True
    return embed_layer


# In[ ]:


class ClassifierModuleForRelation(nn.Module):
    '''
    classifier for both E and E_anti models
    '''
    def __init__(self, args):
        super(ClassifierModuleForRelation, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.pos_embed_layer_e1 = create_relative_pos_embed_layer(args.max_pos_num, args.pos_embedding_dim)
        self.pos_embed_layer_e2 = create_relative_pos_embed_layer(args.max_pos_num, args.pos_embedding_dim)
        
        self.input_dim = args.embedding_dim + args.pos_embedding_dim * 2
        
        self.encoder = RnnModel(self.args, self.input_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.NEG_INF = -1.0e6
        

    def forward(self, word_embeddings, pos_e1, pos_e2, z, mask):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        

        pos_embeddings_e1 = self.pos_embed_layer_e1(pos_e1)
        pos_embeddings_e2 = self.pos_embed_layer_e2(pos_e2)
        embeddings = torch.cat([word_embeddings, pos_embeddings_e1, pos_embeddings_e2], dim=2)
        
        masked_input = embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, mask)
        
        max_hidden = torch.max(hiddens + (1 - mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)

        return predict


# In[ ]:


class IntrospectionGeneratorModuleForRelation(nn.Module):
    '''
    classifier for both E and E_anti models
    '''
    def __init__(self, args):
        super(IntrospectionGeneratorModuleForRelation, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        self.label_embedding_dim = args.label_embedding_dim
        
        self.fixed_classifier = args.fixed_classifier
        
#         self.encoder = RnnModel(self.args, self.input_dim)
#         self.predictor = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.NEG_INF = -1.0e6
        
        self.pos_embed_layer_e1 = create_relative_pos_embed_layer(args.max_pos_num, args.pos_embedding_dim)
        self.pos_embed_layer_e2 = create_relative_pos_embed_layer(args.max_pos_num, args.pos_embedding_dim)
        
        self.input_dim = args.embedding_dim + args.pos_embedding_dim * 2
        
        self.lab_embed_layer = self._create_label_embed_layer() # should be shared with the Classifier_pred weights
        
        # baseline classification model
        self.Classifier_enc = RnnModel(args, self.input_dim)
        self.Classifier_pred = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.Transformation = nn.Sequential()
        self.Transformation.add_module('linear_layer', nn.Linear(self.hidden_dim + self.label_embedding_dim, self.hidden_dim / 2))
        self.Transformation.add_module('tanh_layer', nn.Tanh())
        self.Generator = DepGenerator(args, self.input_dim)
        
        
    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
    
    def forward_cls(self, word_embeddings, pos_e1, pos_e2, mask):
        pos_embeddings_e1 = self.pos_embed_layer_e1(pos_e1)
        pos_embeddings_e2 = self.pos_embed_layer_e2(pos_e2)
        embeddings = torch.cat([word_embeddings, pos_embeddings_e1, pos_embeddings_e2], dim=2)
        
        cls_hiddens = self.Classifier_enc(embeddings, mask) # (batch_size, hidden_dim, sequence_length)
        max_cls_hidden = torch.max(cls_hiddens + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)
        
        cls_pred_logits = self.Classifier_pred(max_cls_hidden) # (batch_size, num_labels)
        
        return cls_pred_logits
    
    
    def forward(self, word_embeddings, pos_e1, pos_e2, mask):
        pos_embeddings_e1 = self.pos_embed_layer_e1(pos_e1)
        pos_embeddings_e2 = self.pos_embed_layer_e2(pos_e2)
        embeddings = torch.cat([word_embeddings, pos_embeddings_e1, pos_embeddings_e2], dim=2)
        
        cls_hiddens = self.Classifier_enc(embeddings, mask) # (batch_size, hidden_dim, sequence_length)
        max_cls_hidden = torch.max(cls_hiddens + (1 - mask).unsqueeze(1) * self.NEG_INF, dim=2)[0] # (batch_size, hidden_dim)
        
        if self.fixed_classifier:
            max_cls_hidden = Variable(max_cls_hidden.data)
        
        cls_pred_logits = self.Classifier_pred(max_cls_hidden) # (batch_size, num_labels)
        
        _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        
        cls_lab_embeddings = self.lab_embed_layer(cls_pred) # (batch_size, lab_emb_dim)
        
        init_h0 = self.Transformation(torch.cat([max_cls_hidden, cls_lab_embeddings], dim=1)) # (batch_size, hidden_dim / 2)
        init_h0 = init_h0.unsqueeze(0).expand(2, init_h0.size(0), init_h0.size(1)).contiguous() # (2, batch_size, hidden_dim / 2)
        
        z_scores_ = self.Generator(word_embeddings, h0=init_h0, mask=mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF
        
        return z_scores_, cls_pred_logits
        


# In[ ]:


class HardIntrospection3PlayerRelationClassificationModel(HardIntrospection3PlayerClassificationModel):
    
    def __init__(self, embeddings, args):
        
        self.pos_embedding_dim = args.pos_embedding_dim
        self.use_relative_pos = args.use_relative_pos
        self.max_pos_num = args.max_pos_num
                
        super(HardIntrospection3PlayerRelationClassificationModel, self).__init__(embeddings, args)
                
        self.generator = IntrospectionGeneratorModuleForRelation(args)
        
        self.E_model = ClassifierModuleForRelation(args)
        self.E_anti_model = ClassifierModuleForRelation(args)
        
    def init_optimizers(self):
        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.args.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.args.lr)
        
    def init_rl_optimizers(self):
        self.opt_G_sup = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr)
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr * 0.1)

    
    def train_cls_one_step(self, x, label, mask, e, pos_e1=None, pos_e2=None):
        
        self.opt_G_sup.zero_grad()
        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        cls_predict = self.generator.forward_cls(word_embeddings, pos_e1, pos_e2, mask)
        
        sup_loss = torch.mean(self.loss_func(cls_predict, label))
        
        losses = {'g_sup_loss':sup_loss.cpu().data}
        
        sup_loss.backward()
        self.opt_G_sup.step()
        
        return losses, cls_predict
    
#     def _get_pos_appended_embedding(self, word_embeddings, e, pos_e1, pos_e2):
#         if self.pos_embedding_dim > 0:
#             if self.use_relative_pos:
#                 pos_embeddings_e1 = self.pos_embed_layer_e1(pos_e1)
#                 pos_embeddings_e2 = self.pos_embed_layer_e2(pos_e2)
#                 embeddings = torch.cat([word_embeddings, pos_embeddings_e1, pos_embeddings_e2], dim=2)
#             else:
#                 pos_embeddings = self.pos_embed_layer(e) #(batch_size, length, 3)
#                 embeddings = torch.cat([word_embeddings, pos_embeddings], dim=2)
#         else:
#             embeddings = word_embeddings
#         return embeddings
    
    def forward_cls(self, x, e, mask, pos_e1=None, pos_e2=None):
    
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        cls_predict = self.generator.forward_cls(word_embeddings, pos_e1, pos_e2, mask)

        return cls_predict
    
    
    def train_one_step(self, x, label, baseline, mask, e, pos_e1=None, pos_e2=None):
        
        # TODO: try to see whether removing the follows makes any differences
        self.opt_E_anti.zero_grad()
        self.opt_E.zero_grad()
        self.opt_G_sup.zero_grad()
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, cls_predict, z, neg_log_probs = self.forward(x, e, mask, pos_e1, pos_e2)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
#         e_loss = torch.mean(self.loss_func(predict, label))
        _, cls_pred = torch.max(cls_predict, dim=1) # (batch_size,)
#         e_loss = torch.mean(self.loss_func(predict, cls_pred)) # e_loss comes from only consistency
        e_loss = (torch.mean(self.loss_func(predict, label)) + torch.mean(self.loss_func(predict, cls_pred))) / 2
        
        # g_sup_loss comes from only cls pred loss
        g_sup_loss, g_rl_loss, rewards, consistency_loss, continuity_loss, sparsity_loss = self.get_loss(predict, 
                                                                         anti_predict, 
                                                                         cls_predict, label, z, 
                                                                         neg_log_probs, baseline, mask)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_sup_loss':g_sup_loss.cpu().data, 'g_rl_loss':g_rl_loss.cpu().data}
        
#         if self.game_mode == '3player' and not self.fixed_E_anti:
        if self.game_mode == '3player' or self.game_mode == 'taos':
            e_loss_anti.backward(retain_graph=True)
            self.opt_E_anti.step()
            self.opt_E_anti.zero_grad()
        
        e_loss.backward(retain_graph=True)
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        if not self.fixed_classifier:
            g_sup_loss.backward(retain_graph=True)
            self.opt_G_sup.step()
            self.opt_G_sup.zero_grad()

        g_rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, cls_predict, z, rewards, consistency_loss, continuity_loss, sparsity_loss  
    
    def forward(self, x, e, mask, pos_e1=None, pos_e2=None):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        z_scores_, cls_predict = self.generator(word_embeddings, pos_e1, pos_e2, mask)
        
        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_) #(batch_size, length)
        
        predict = self.E_model(word_embeddings, pos_e1, pos_e2, z, mask)
        
        e1_mask = (pos_e1 != 0).type(torch.FloatTensor).cuda()
        e2_mask = (pos_e2 != 0).type(torch.FloatTensor).cuda()
#         masked_word_embeddings = word_embeddings * (1 - z * e1_mask * e2_mask).unsqueeze(-1)
#         masked_embeddings = torch.cat([masked_word_embeddings, pos_embeddings_e1, pos_embeddings_e2], dim=2)
#         anti_predict = self.E_anti_model(masked_embeddings, torch.ones_like(z), mask)
        anti_predict = self.E_anti_model(word_embeddings, pos_e1, pos_e2, 1 - z * e1_mask * e2_mask, mask)

        return predict, anti_predict, cls_predict, z, neg_log_probs
    
    
    def get_advantages(self, pred_logits, anti_pred_logits, cls_pred_logits, label, z, neg_log_probs, baseline, mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # supervised loss
        prediction_loss = self.loss_func(cls_pred_logits, label) # (batch_size, )
        sup_loss = torch.mean(prediction_loss)
        
        # total loss of accuracy (not batchwise)
        _, cls_pred = torch.max(cls_pred_logits, dim=1) # (batch_size,)
        _, ver_pred = torch.max(pred_logits, dim=1) # (batch_size,)
        consistency_loss = self.loss_func(pred_logits, cls_pred)
        
        if self.fixed_classifier == False:
            prediction = (cls_pred == label).type(torch.FloatTensor)
            pred_consistency = (ver_pred == cls_pred).type(torch.FloatTensor)
        else:
            prediction = (ver_pred == label).type(torch.FloatTensor)
            pred_consistency = (ver_pred == cls_pred).type(torch.FloatTensor)
        
        _, anti_pred = torch.max(anti_pred_logits, dim=1)
        prediction_anti = (anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            pred_consistency = pred_consistency.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        continuity_loss, sparsity_loss = count_regularization_hinge_loss_batch(z, self.highlight_count, mask)
#         continuity_loss, sparsity_loss = count_regularization_loss_batch(z, self.highlight_count, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward 
#         rewards = (prediction + pred_consistency) * self.args.lambda_pos_reward - prediction_anti - sparsity_loss - continuity_loss
        if self.game_mode.startswith('3player'):
#             rewards = 0.5 * ((prediction + pred_consistency) / 2 - prediction_anti) - sparsity_loss - continuity_loss
#             rewards = 0.5 * ((prediction + pred_consistency) - prediction_anti) - sparsity_loss - continuity_loss
            rewards = (prediction + pred_consistency) - self.lambda_acc_gap * prediction_anti - sparsity_loss - continuity_loss
        else:
            rewards = prediction + pred_consistency - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()
        
        return sup_loss, advantages, rewards, pred_consistency, continuity_loss, sparsity_loss

