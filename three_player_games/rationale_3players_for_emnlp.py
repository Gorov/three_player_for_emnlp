
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

from rationale_3players_sentence_classification_models import ClassifierModule, HardRationale3PlayerClassificationModel
from models.language_models import NGramLanguageModeler

from collections import deque


# In[ ]:


def count_regularization_baos_for_both(z, count_tokens, count_pieces, mask=None):
    """
    Compute regularization loss, based on a given rationale sequence
    Use Yujia's formulation

    Inputs:
        z -- torch variable, "binary" rationale, (batch_size, sequence_length)
        percentage -- the percentage of words to keep
    Outputs:
        a loss value that contains two parts:
        continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
        sparsity_loss -- |mean(z_{i}) - percent|
    """

    # (batch_size,)
    if mask is not None:
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)
    else:
        mask_z = z
        seq_lengths = torch.sum(z - z + 1.0, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
    percentage = count_pieces * 2 / seq_lengths
#     continuity_loss = F.threshold(continuity_ratio - percentage, 0, 0, False)
    continuity_loss = torch.abs(continuity_ratio - percentage)
    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    percentage = count_tokens / seq_lengths #(batch_size,)
#     sparsity_loss = F.threshold(sparsity_ratio - percentage, 0, 0, False)
    sparsity_loss = torch.abs(sparsity_ratio - percentage)

    return continuity_loss, sparsity_loss


# In[ ]:


class HardRationale3PlayerClassificationModelForEmnlp(HardRationale3PlayerClassificationModel):
    
    def __init__(self, embeddings, args):
        super(HardRationale3PlayerClassificationModelForEmnlp, self).__init__(embeddings, args)
        self.game_mode = args.game_mode
        self.ngram = args.ngram
        
        self.lambda_acc_gap = args.lambda_acc_gap
        
        if self.game_mode == '3player_self_play':
            self.E_anti_model = self.E_model
        elif self.game_mode == '3player_fixed_d':
            pass
        elif self.game_mode == '3player_relaxed_d':
            pass

    
    def train_one_step(self, x, label, baseline, mask, with_lm=False):
        
        predict, anti_predict, z, neg_log_probs = self.forward(x, mask)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, z, 
                                                                     neg_log_probs, baseline, 
                                                                     mask, label)
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        if self.game_mode == '3player' or self.game_mode == 'taos': 
        # taos only trains E_anti, but not use it to affect the reward
            e_loss_anti.backward()
            self.opt_E_anti.step()
            self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, z, rewards, continuity_loss, sparsity_loss
    
    
    def train_cls_one_step(self, x, label, mask):
        
        predict = self.forward_cls(x, mask)
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        losses = {'e_loss':e_loss.cpu().data}
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        return losses, predict
    
    
    def forward_cls(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        z = torch.ones_like(x).type(torch.cuda.FloatTensor)
        
        predict = self.E_model(word_embeddings, z, mask)

        return predict
    
        
    def forward(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        neg_inf = -1.0e6
        
        z_scores_ = self.generator(word_embeddings, mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_)
        
        predict = self.E_model(word_embeddings, z, mask)
        
        anti_predict = self.E_anti_model(word_embeddings, 1 - z, mask)

        return predict, anti_predict, z, neg_log_probs

    
    def get_advantages(self, predict, anti_predict, label, z, neg_log_probs, baseline, mask, x=None):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        _, y_pred = torch.max(predict, dim=1)
        prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
#         continuity_loss, sparsity_loss = bao_regularization_loss_batch(z, self.highlight_percentage, mask)
        continuity_loss, sparsity_loss = count_regularization_baos_for_both(z, self.count_tokens, self.count_pieces, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward
        if self.game_mode.startswith('3player'):
#             rewards = prediction - prediction_anti - sparsity_loss - continuity_loss
            rewards = 0.1 * prediction + self.lambda_acc_gap * (prediction - prediction_anti) - sparsity_loss - continuity_loss
        else:
            rewards = prediction - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()
        
        if x is None:
            return advantages, rewards, continuity_loss, sparsity_loss
    
#     def large_margin_losses(enc_logits, dis_logits, labels, margin):
#         """
#         This function calculates:
#         1) generator large-margin loss
#         2) encoder cross_entropy loss
#         3) discriminator cross_entropy loss

#         logits -- (batch_size, num_classes)
#         labels -- need to be one-hot / soft one-hot.
#                   (batch_size, num_classes)    

#         """
#         # encoder losses -- (batch_size, )
#         encoder_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#             logits=enc_logits, labels=labels) 
#         # discriminator losses -- (batch_size, )
#         dis_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
#             logits=dis_logits, labels=labels)

#         # generator large margin loss
#         # generator_losses = tf.maximum(dis_losses - encoder_losses, margin)
#         generator_losses = tf.maximum(margin - (dis_losses - encoder_losses), 0)
#         gen_margin_loss = tf.reduce_mean(generator_losses)
#         enc_loss = tf.reduce_mean(encoder_losses)
#         dis_loss = tf.reduce_mean(dis_losses)

#         return gen_margin_loss, enc_loss, dis_loss
    
    def get_loss(self, predict, anti_predict, z, neg_log_probs, baseline, mask, label, x=None):
        reward_tuple = self.get_advantages(predict, anti_predict, label, z, neg_log_probs, baseline, mask, x)
        
        if x is None:
            advantages, rewards, continuity_loss, sparsity_loss = reward_tuple
        else:
            advantages, rewards, continuity_loss, sparsity_loss, lm_prob = reward_tuple
        
        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        if x is None:
            return rl_loss, rewards, continuity_loss, sparsity_loss
        else:
            return rl_loss, rewards, continuity_loss, sparsity_loss, lm_prob
        

