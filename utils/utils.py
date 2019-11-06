
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[ ]:


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    
    def forward(self, x):
        """
        transpose dim1 and dim2 of input tensor x
        """
        return x.transpose(self.dim1, self.dim2).contiguous()


# In[ ]:


def single_regularization_loss_batch(z, mask):
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
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)
    sparsity_loss = torch.sum(mask_z, dim=-1) / seq_lengths  #(batch_size,)

    return sparsity_loss, continuity_loss


# In[ ]:


def bao_regularization_loss_batch(z, percentage, mask=None):
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
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)
    sparsity_loss = torch.abs(torch.sum(mask_z, dim=-1) / seq_lengths - percentage)  #(batch_size,)

    return continuity_loss, sparsity_loss


# In[ ]:


def bao_regularization_hinge_loss_batch(z, percentage, mask=None):
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
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    
    sparsity_loss = F.threshold(sparsity_ratio - percentage, 0, 0, False)

    return continuity_loss, sparsity_loss


# In[ ]:


def count_regularization_loss_batch(z, count, mask=None):
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
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    
    percentage = count / seq_lengths #(batch_size,)
    
    sparsity_loss = torch.abs(sparsity_ratio - percentage)  #(batch_size,)

    return continuity_loss, sparsity_loss


# In[ ]:


def count_regularization_hinge_loss_batch(z, count, mask=None):
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
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    
    percentage = count / seq_lengths #(batch_size,)
    
    sparsity_loss = F.threshold(sparsity_ratio - percentage, 0, 0, False)

    return continuity_loss, sparsity_loss


# In[ ]:


def bao_regularization_hinge_loss_batch_with_none_loss(z, percentage, none_relation, mask):
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
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    
    sparsity_loss = F.threshold(sparsity_ratio - percentage, 0, 0, False) * (1 - none_relation) +                        sparsity_ratio * none_relation

    return continuity_loss, sparsity_loss


# In[ ]:


def show_1d_rationale(probs, text, max_len=50):
    """
    show 1d bar plot. 
        probs -- a 1D rationale probs
        text -- the string of the text
    """
    plt.figure(num=99, figsize=(12, 5))
    
    if len(text) < max_len:
        show_len = len(text)
    else:
        show_len = max_len
    plt.bar(range(show_len), probs[0:show_len])
    
    plt.xticks(range(show_len), text[0:show_len], size='small', rotation="vertical")     
    plt.title('Probability distribution of the first %d words' %(max_len))
    plt.show()
    
    


# In[ ]:




