
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

from collections import deque


# In[ ]:


class MatchingClassifierModule(nn.Module):
    '''
    classifier for both E and E_anti models
    '''
    def __init__(self, args):
        super(MatchingClassifierModule, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.input_dim = args.embedding_dim
        
        if self.args.dropout > 0:
            self.dropout_layer = nn.Dropout(self.args.dropout)
        
        self.encoder = RnnModel(self.args, self.input_dim)
        self.predictor = nn.Linear(self.hidden_dim * 4, self.num_labels)
        
        self.NEG_INF = -1.0e6
        

    def forward(self, q_embeddings, p_embeddings, z_q, z_p, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        

        q_masked_input = q_embeddings * z_q.unsqueeze(-1)
        q_hiddens = self.encoder(q_masked_input, q_mask)
        
        p_masked_input = p_embeddings * z_p.unsqueeze(-1)
        p_hiddens_sort_ = self.encoder(p_masked_input[p_sort_idx,:,:], p_mask[p_sort_idx,:])
        p_hiddens = p_hiddens_sort_[revert_p_idx, :, :]
        
        if self.args.dropout > 0:
            q_hiddens = self.dropout_layer(q_hiddens)
            p_hiddens = self.dropout_layer(p_hiddens)
        
        q_max_hidden = torch.max(q_hiddens + (1 - q_mask * z_q).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
#         print p_embeddings.size()
#         print p_masked_input.size()
#         print p_hiddens_sort_.size()
#         print p_hiddens.size()
#         print p_mask.size()
#         print z_p.size()
        p_max_hidden = torch.max(p_hiddens + (1 - p_mask * z_p).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
#         print(q_mask.size())
#         print(z_q.size())
#         print(q_hiddens.size())
    
#         q_max_hidden = torch.sum(q_hiddens * (q_mask * z_q).unsqueeze(1), dim=2) / torch.sum(q_mask * z_q, dim=1).unsqueeze(1)
#         p_max_hidden = torch.sum(p_hiddens * (p_mask * z_p).unsqueeze(1), dim=2) / torch.sum(p_mask * z_p, dim=1).unsqueeze(1)
        
        predict = self.predictor(torch.cat([q_max_hidden, p_max_hidden, q_max_hidden * p_max_hidden, 
                                            torch.abs(q_max_hidden - p_max_hidden)], dim=1))

        return predict


# In[ ]:


class IntrospectionGeneratorModule(nn.Module):
    def __init__(self, args):
        super(IntrospectionGeneratorModule, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        self.label_embedding_dim = args.label_embedding_dim
        
        self.fixed_classifier = args.fixed_classifier
        
        self.input_dim = args.embedding_dim
        
        self.NEG_INF = -1.0e6
        
        self.lab_embed_layer = self._create_label_embed_layer() # should be shared with the Classifier_pred weights
        
        # baseline classification model
        self.Transformation = nn.Sequential()
        self.Transformation.add_module('linear_layer', nn.Linear(self.label_embedding_dim, self.hidden_dim / 2))
        self.Transformation.add_module('tanh_layer', nn.Tanh())
        self.Generator = DepGenerator(args, self.input_dim)
        
        
    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.label_embedding_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer
    
    
    def forward(self, word_embeddings, cls_pred, mask):
        
        cls_lab_embeddings = self.lab_embed_layer(cls_pred) # (batch_size, lab_emb_dim)
        
        init_h0 = self.Transformation(cls_lab_embeddings) # (batch_size, hidden_dim / 2)
        init_h0 = init_h0.unsqueeze(0).expand(2, init_h0.size(0), init_h0.size(1)).contiguous() # (2, batch_size, hidden_dim / 2)
        
        z_scores_ = self.Generator(word_embeddings, h0=init_h0, mask=mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF
        
        return z_scores_
        


# In[ ]:


class Rationale3PlayerMatchingModel(nn.Module):
    
    def __init__(self, embeddings, args):
        super(Rationale3PlayerMatchingModel, self).__init__()
        self.args = args
        
        self.model_type = args.model_type
        self.use_cuda = args.cuda
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        
        self.NEG_INF = -1.0e6
                    
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.input_dim = args.embedding_dim
        
        self.E_model = MatchingClassifierModule(args)
        self.E_anti_model = MatchingClassifierModule(args)
        
        self.loss_func = nn.CrossEntropyLoss()
        
        
    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = self.args.fine_tuning
        return embed_layer
        
    def forward(self, x, mask):
        pass
    


# In[ ]:


class HardRationale3PlayerMatchingModel(Rationale3PlayerMatchingModel):
    
    def __init__(self, embeddings, args):
        super(HardRationale3PlayerMatchingModel, self).__init__(embeddings, args)
        self.generator = Generator(args, self.input_dim)
        self.highlight_percentage = args.highlight_percentage
        self.highlight_count = args.highlight_count
        self.exploration_rate = args.exploration_rate
        
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        self.game_mode = args.game_mode
        
        if args.margin is not None:
            self.margin = args.margin

        
    def init_optimizers(self):
        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.args.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.args.lr)
        
    def init_rl_optimizers(self):
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr * 0.1)
        
        
    def init_reward_queue(self):
        queue_length = 200
        self.z_q_history_rewards = deque(maxlen=queue_length)
        self.z_q_history_rewards.append(0.)
        
        self.z_p_history_rewards = deque(maxlen=queue_length)
        self.z_p_history_rewards.append(0.)
        
                
    def _generate_rationales(self, z_prob_):
        '''
        Input:
            z_prob_ -- (num_rows, length, 2)
        Output:
            z -- (num_rows, length)
        '''        
        z_prob__ = z_prob_.view(-1, 2) # (num_rows * length, 2)
        
        # sample actions
        sampler = torch.distributions.Categorical(z_prob__)
        if self.training:
            z_ = sampler.sample() # (num_rows * p_length,)
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]
        
        #(num_rows, length)
        z = z_.view(z_prob_.size(0), z_prob_.size(1))
        
        if self.use_cuda == True:
            z = z.type(torch.cuda.FloatTensor)
        else:
            z = z.type(torch.FloatTensor)
            
        # (num_rows * length,)
        neg_log_probs_ = -sampler.log_prob(z_)
        # (num_rows, length)
        neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))
        
        return z, neg_log_probs
    
    
    def train_cls_one_step(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        
        self.opt_E.zero_grad()
        self.opt_E_anti.zero_grad()
        
        predict = self.forward_cls(q, p, q_mask, p_mask, p_sort_idx, revert_p_idx)
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        losses = {'e_loss':e_loss.cpu().data}

        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        return losses, predict
    
    
    def train_gen_one_step(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
        
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask,
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
#         losses = {'g_rl_loss':rl_loss.cpu().data}
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
    
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss

    
    def train_one_step(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
            
        self.opt_G_rl.zero_grad()
        self.opt_E.zero_grad()
        self.opt_E_anti.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask, 
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss
    
    
    def train_one_step_predictors(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
            
        self.opt_G_rl.zero_grad()
        self.opt_E.zero_grad()
        self.opt_E_anti.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask, 
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
#         losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data}
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss

    
    def forward_cls(self, q, p, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        q_embeddings = self.embed_layer(q) #(batch_size, length, embedding_dim)
        p_embeddings = self.embed_layer(p) #(batch_size, length, embedding_dim)
        
        neg_inf = -1.0e6
        
        z_q = torch.ones_like(q_mask)
        z_p = torch.ones_like(p_mask)
        
        predict = self.E_model(q_embeddings, p_embeddings, z_q, z_p, q_mask, p_mask, p_sort_idx, revert_p_idx)

        return predict
    
        
    def forward(self, q, p, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        q_embeddings = self.embed_layer(q) #(batch_size, length, embedding_dim)
        p_embeddings = self.embed_layer(p) #(batch_size, length, embedding_dim)
        
        neg_inf = -1.0e6
        
        z_scores_ = self.generator(q_embeddings, q_mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - q_mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (q_mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - q_mask.unsqueeze(-1)) * z_probs_)
        
        z_q, q_neg_log_probs = self._generate_rationales(z_probs_)
        
        z_scores_sort_ = self.generator(p_embeddings[p_sort_idx,:,:], p_mask[p_sort_idx,:])
        z_scores_ = z_scores_sort_[revert_p_idx, :, :]
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - p_mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (p_mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - p_mask.unsqueeze(-1)) * z_probs_)
        
        z_p, p_neg_log_probs = self._generate_rationales(z_probs_)
        
        predict = self.E_model(q_embeddings, p_embeddings, z_q, z_p, q_mask, p_mask, p_sort_idx, revert_p_idx)
        
        anti_predict = self.E_anti_model(q_embeddings, p_embeddings, 1 - z_q, 1 - z_p, q_mask, p_mask, p_sort_idx, revert_p_idx)

        return predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs

    
    def get_advantages(self, predict, anti_predict, label, z_q, z_p, 
                       q_neg_log_probs, p_neg_log_probs, q_baseline, p_baseline, q_mask, p_mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        _, y_pred = torch.max(predict, dim=1)
        if self.game_mode.startswith('3player'):
            prediction = (y_pred == label).type(torch.FloatTensor) * (self.lambda_anti + 0.2)
#             prediction = (y_pred == label).type(torch.FloatTensor) * 0.2
        else:
            prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        q_continuity_loss, q_sparsity_loss = bao_regularization_loss_batch(z_q, self.highlight_percentage, q_mask)
        p_continuity_loss, p_sparsity_loss = bao_regularization_loss_batch(z_p, self.highlight_percentage, p_mask)
#         continuity_loss, sparsity_loss = bao_regularization_hinge_loss_batch(z, self.highlight_percentage, mask)
#         continuity_loss, sparsity_loss = count_regularization_hinge_loss_batch(z, self.highlight_count, mask)
#         continuity_loss, sparsity_loss = bao_regularization_hinge_loss_batch_with_none_loss(z, self.highlight_percentage, 
#                                                                              self.none_relation_id, mask)

        q_continuity_loss = q_continuity_loss * self.lambda_continuity
        p_continuity_loss = p_continuity_loss * self.lambda_continuity
        q_sparsity_loss = q_sparsity_loss * self.lambda_sparsity
        p_sparsity_loss = p_sparsity_loss * self.lambda_sparsity

        # batch RL reward
        if self.game_mode.startswith('3player'):
            q_rewards = prediction - prediction_anti - q_sparsity_loss - q_continuity_loss
            p_rewards = prediction - prediction_anti - p_sparsity_loss - p_continuity_loss
        else:
            q_rewards = prediction - q_sparsity_loss - q_continuity_loss
            p_rewards = prediction - p_sparsity_loss - p_continuity_loss
        
        q_advantages = q_rewards - q_baseline # (batch_size,)
        q_advantages = Variable(q_advantages.data, requires_grad=False)
        if self.use_cuda:
            q_advantages = q_advantages.cuda()
            
        p_advantages = p_rewards - p_baseline # (batch_size,)
        p_advantages = Variable(p_advantages.data, requires_grad=False)
        if self.use_cuda:
            p_advantages = p_advantages.cuda()
        
        return q_advantages, p_advantages, q_rewards, p_rewards, q_continuity_loss, q_sparsity_loss
    
    def get_listwise_advantages(self, predict, anti_predict, label, z_q, z_p, 
                       q_neg_log_probs, p_neg_log_probs, q_baseline, p_baseline, q_mask, p_mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        
        # predict --  (batch * sample, 2) -> (batch, sample) -> soft
        
        predict_2d = predict[:,1].contiguous().view(self.args.batch_size, -1)
        anti_predict_2d = anti_predict[:,1].contiguous().view(self.args.batch_size, -1)
        
        _, y_pred = torch.max(predict_2d, dim=1)
        prediction = (y_pred == 0).type(torch.FloatTensor) * (self.lambda_anti + 0.2)
        prediction = prediction.unsqueeze(1).expand_as(predict_2d).contiguous().view(predict.size(0))
        
        _, y_anti_pred = torch.max(anti_predict_2d, dim=1)
        prediction_anti = (y_anti_pred == 0).type(torch.FloatTensor) * self.lambda_anti
        prediction_anti = prediction_anti.unsqueeze(1).expand_as(anti_predict_2d).contiguous().view(anti_predict.size(0))
        
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        q_continuity_loss, q_sparsity_loss = bao_regularization_loss_batch(z_q, self.highlight_percentage, q_mask)
        p_continuity_loss, p_sparsity_loss = bao_regularization_loss_batch(z_p, self.highlight_percentage, p_mask)

        q_continuity_loss = q_continuity_loss * self.lambda_continuity
        p_continuity_loss = p_continuity_loss * self.lambda_continuity
        q_sparsity_loss = q_sparsity_loss * self.lambda_sparsity
        p_sparsity_loss = p_sparsity_loss * self.lambda_sparsity

        # batch RL reward
        if self.game_mode.startswith('3player'):
            q_rewards = prediction - prediction_anti - q_sparsity_loss - q_continuity_loss
            p_rewards = prediction - prediction_anti - p_sparsity_loss - p_continuity_loss
        else:
            q_rewards = prediction - q_sparsity_loss - q_continuity_loss
            p_rewards = prediction - p_sparsity_loss - p_continuity_loss
        
        q_advantages = q_rewards - q_baseline # (batch_size,)
        q_advantages = Variable(q_advantages.data, requires_grad=False)
        if self.use_cuda:
            q_advantages = q_advantages.cuda()
            
        p_advantages = p_rewards - p_baseline # (batch_size,)
        p_advantages = Variable(p_advantages.data, requires_grad=False)
        if self.use_cuda:
            p_advantages = p_advantages.cuda()
        
        return q_advantages, p_advantages, q_rewards, p_rewards, q_continuity_loss, q_sparsity_loss
    
    def get_loss(self, predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs, 
                 q_baseline, p_baseline, q_mask, p_mask, label):
        reward_tuple = self.get_advantages(predict, anti_predict, label, z_q, z_p, 
                                           q_neg_log_probs, p_neg_log_probs, 
                                           q_baseline, p_baseline, q_mask, p_mask)
#         reward_tuple = self.get_listwise_advantages(predict, anti_predict, label, z_q, z_p, 
#                                            q_neg_log_probs, p_neg_log_probs, 
#                                            q_baseline, p_baseline, q_mask, p_mask)
        
        q_advantages, p_advantages, q_rewards, p_rewards, continuity_loss, sparsity_loss = reward_tuple
        
        # (batch_size, q_length)
        q_advantages_expand_ = q_advantages.unsqueeze(-1).expand_as(q_neg_log_probs)
        p_advantages_expand_ = p_advantages.unsqueeze(-1).expand_as(p_neg_log_probs)
        q_rl_loss = torch.sum(q_neg_log_probs * q_advantages_expand_ * q_mask)
        p_rl_loss = torch.sum(p_neg_log_probs * p_advantages_expand_ * p_mask)
        
        rl_loss = (q_rl_loss + p_rl_loss) / 2
        
        return rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss


# In[ ]:


class HardIntrospection3PlayerMatchingModel(HardRationale3PlayerMatchingModel):
    
    def __init__(self, embeddings, args):
        super(HardIntrospection3PlayerMatchingModel, self).__init__(embeddings, args)
        self.generator = IntrospectionGeneratorModule(args)
        self.classifier = MatchingClassifierModule(args)
    
    
    def train_gen_one_step(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
        
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask,
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
#         losses = {'g_rl_loss':rl_loss.cpu().data}
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
    
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss

    
    def train_one_step(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
            
        self.opt_G_rl.zero_grad()
        self.opt_E.zero_grad()
        self.opt_E_anti.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask, 
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss
    
    
    def train_one_step_predictors(self, q, p, label, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        z_q_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_q_history_rewards))]))
        if self.args.cuda:
            z_q_baseline = z_q_baseline.cuda()
            
        z_p_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_p_history_rewards))]))
        if self.args.cuda:
            z_p_baseline = z_p_baseline.cuda()
            
        self.opt_G_rl.zero_grad()
        self.opt_E.zero_grad()
        self.opt_E_anti.zero_grad()
        
        predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs = self.forward(q, p, q_mask, p_mask, 
                                                                                         p_sort_idx, revert_p_idx)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, 
                                                                         z_q, z_p, q_neg_log_probs, p_neg_log_probs,
                                                                         z_q_baseline, z_p_baseline, 
                                                                         q_mask, p_mask, label)
        
#         losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data}
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()
        
        z_q_batch_reward = np.mean(q_rewards.cpu().data.numpy())
        self.z_q_history_rewards.append(z_q_batch_reward)
        
        z_p_batch_reward = np.mean(p_rewards.cpu().data.numpy())
        self.z_p_history_rewards.append(z_p_batch_reward)
        
        rewards = (q_rewards + p_rewards) / 2
        
        return losses, predict, anti_predict, z_q, z_p, rewards, continuity_loss, sparsity_loss
    
        
    def forward(self, q, p, q_mask, p_mask, p_sort_idx=None, revert_p_idx=None):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """
        
        q_embeddings = self.embed_layer(q) #(batch_size, length, embedding_dim)
        p_embeddings = self.embed_layer(p) #(batch_size, length, embedding_dim)
        
        z_q_ = torch.ones_like(q_mask)
        z_p_ = torch.ones_like(p_mask)
        
        cls_predict = self.classifier(q_embeddings, p_embeddings, z_q_, z_p_, q_mask, p_mask, p_sort_idx, revert_p_idx)
        _, cls_predict = torch.max(cls_predict, dim=1) # (batch_size,)
        
        neg_inf = -1.0e6
        
        z_scores_ = self.generator(q_embeddings, cls_predict, q_mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - q_mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (q_mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - q_mask.unsqueeze(-1)) * z_probs_)
        
        z_q, q_neg_log_probs = self._generate_rationales(z_probs_)
        
        z_scores_sort_ = self.generator(p_embeddings[p_sort_idx,:,:], cls_predict, p_mask[p_sort_idx,:])
        z_scores_ = z_scores_sort_[revert_p_idx, :, :]
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - p_mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (p_mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - p_mask.unsqueeze(-1)) * z_probs_)
        
        z_p, p_neg_log_probs = self._generate_rationales(z_probs_)
        
        predict = self.E_model(q_embeddings, p_embeddings, z_q, z_p, q_mask, p_mask, p_sort_idx, revert_p_idx)
        
        anti_predict = self.E_anti_model(q_embeddings, p_embeddings, 1 - z_q, 1 - z_p, q_mask, p_mask, p_sort_idx, revert_p_idx)

        return predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs

    
    def get_advantages(self, predict, anti_predict, label, z_q, z_p, 
                       q_neg_log_probs, p_neg_log_probs, q_baseline, p_baseline, q_mask, p_mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        _, y_pred = torch.max(predict, dim=1)
        if self.game_mode.startswith('3player'):
            prediction = (y_pred == label).type(torch.FloatTensor) * (self.lambda_anti + 0.2)
#             prediction = (y_pred == label).type(torch.FloatTensor) * 0.2
#             prediction = (y_pred == label).type(torch.FloatTensor)
        else:
            prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        q_continuity_loss, q_sparsity_loss = bao_regularization_loss_batch(z_q, self.highlight_percentage, q_mask)
        p_continuity_loss, p_sparsity_loss = bao_regularization_loss_batch(z_p, self.highlight_percentage, p_mask)

        q_continuity_loss = q_continuity_loss * self.lambda_continuity
        p_continuity_loss = p_continuity_loss * self.lambda_continuity
        q_sparsity_loss = q_sparsity_loss * self.lambda_sparsity
        p_sparsity_loss = p_sparsity_loss * self.lambda_sparsity

        # batch RL reward
        if self.game_mode.startswith('3player'):
#             q_rewards = prediction - prediction_anti - q_sparsity_loss - q_continuity_loss
#             p_rewards = prediction - prediction_anti - p_sparsity_loss - p_continuity_loss
            q_rewards = - prediction_anti - q_sparsity_loss - q_continuity_loss
            p_rewards = - prediction_anti - p_sparsity_loss - p_continuity_loss
        else:
            q_rewards = prediction - q_sparsity_loss - q_continuity_loss
            p_rewards = prediction - p_sparsity_loss - p_continuity_loss
        
        q_advantages = q_rewards - q_baseline # (batch_size,)
        q_advantages = Variable(q_advantages.data, requires_grad=False)
        if self.use_cuda:
            q_advantages = q_advantages.cuda()
            
        p_advantages = p_rewards - p_baseline # (batch_size,)
        p_advantages = Variable(p_advantages.data, requires_grad=False)
        if self.use_cuda:
            p_advantages = p_advantages.cuda()
        
        return q_advantages, p_advantages, q_rewards, p_rewards, q_continuity_loss, q_sparsity_loss
    
    
    def get_loss(self, predict, anti_predict, z_q, z_p, q_neg_log_probs, p_neg_log_probs, 
                 q_baseline, p_baseline, q_mask, p_mask, label):
        reward_tuple = self.get_advantages(predict, anti_predict, label, z_q, z_p, 
                                           q_neg_log_probs, p_neg_log_probs, 
                                           q_baseline, p_baseline, q_mask, p_mask)
#         reward_tuple = self.get_listwise_advantages(predict, anti_predict, label, z_q, z_p, 
#                                            q_neg_log_probs, p_neg_log_probs, 
#                                            q_baseline, p_baseline, q_mask, p_mask)
        
        q_advantages, p_advantages, q_rewards, p_rewards, continuity_loss, sparsity_loss = reward_tuple
        
        # (batch_size, q_length)
        q_advantages_expand_ = q_advantages.unsqueeze(-1).expand_as(q_neg_log_probs)
        p_advantages_expand_ = p_advantages.unsqueeze(-1).expand_as(p_neg_log_probs)
        q_rl_loss = torch.sum(q_neg_log_probs * q_advantages_expand_ * q_mask)
        p_rl_loss = torch.sum(p_neg_log_probs * p_advantages_expand_ * p_mask)
        
        rl_loss = (q_rl_loss + p_rl_loss) / 2
        
        return rl_loss, q_rewards, p_rewards, continuity_loss, sparsity_loss

