
# coding: utf-8

# In[ ]:


import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy, random, sys, os

import tqdm


# In[ ]:


def _get_sparsity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    sparsity_count = torch.sum(mask_z, dim=-1)

    return sparsity_ratio, sparsity_count

def _get_continuity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
    continuity_count = torch.sum(torch.abs(mask_z - mask_z_), dim=-1)
    
    return continuity_ratio, continuity_count

def _get_pos_neg_sparsity(z, mask, none_relation):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    
    pos_sparsity_ratio = sparsity_ratio * (1 - none_relation)
    neg_sparsity_ratio = sparsity_ratio * none_relation
    pos_count = (1 - none_relation).sum()
    neg_count = none_relation.sum()

    return pos_sparsity_ratio, neg_sparsity_ratio, pos_count, neg_count


# In[ ]:


def copy_classifier_module(trg_module, enc_path, pred_path):
    trg_module.encoder = torch.load(enc_path)
    trg_module.predictor = torch.load(pred_path)


# In[ ]:


# def evaluate_rationale_model(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, eval_accs, print_train_flag=True):
#     classification_model.eval()
#     eval_sets = ['dev']
#     for set_id, set_name in enumerate(eval_sets):
#         output = ''
#         output2 = ''
#         start_id = 8001
#         dev_correct = 0.0
#         dev_anti_correct = 0.0
#         dev_cls_correct = 0.0
#         dev_total = 0.0
#         sparsity_total = 0.0
#         dev_count = 0

#         pos_sparsity_total = 0.0
#         neg_sparsity_total = 0.0
#         pos_dev_count = 0
#         neg_dev_count = 0

#         num_dev_instance = beer_data.data_sets[set_name].size()

#         for start in range(num_dev_instance / args.batch_size):
#             x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
#                                              sort=True)

#             batch_x_ = Variable(torch.from_numpy(x_mat))
#             batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
#             batch_y_ = Variable(torch.from_numpy(y_vec))
#             if args.cuda:
#                 batch_x_ = batch_x_.cuda()
#                 batch_m_ = batch_m_.cuda()
#                 batch_y_ = batch_y_.cuda()

#             predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)

#             # calculate classification accuarcy
#             _, y_pred = torch.max(predict, dim=1)
#             _, anti_y_pred = torch.max(anti_predict, dim=1)
#             _, y_cls_pred = torch.max(cls_predict, dim=1)

#             dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
#             dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
#             dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
#             dev_total += args.batch_size

#             if len(z.shape) == 3:
#                 mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
#                 mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
#                 sparsity_ratio, sparsity_count = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
#                 sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
#             else:
#                 sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
#                 sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
#                 continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
#                 continuity_total += continuity_ratio.cpu().data.numpy().sum() 
#             dev_count += args.batch_size

#         if set_name == 'dev':
#             dev_accs.append(dev_correct / dev_total)
#             dev_anti_accs.append(dev_anti_correct / dev_total)
#             dev_cls_accs.append(dev_cls_correct / dev_total)
#             if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
#                 best_dev_acc = dev_correct / dev_total

#         else:
#             test_accs.append(dev_correct / dev_total)

# #     if (i+1) % display_iteration == 0:
#     if print_train_flag:
#         print('train:', train_accs[-1], 'consistency_loss:', torch.mean(consistency_loss).cpu().data[0], 
#               'sparsity_loss:', torch.mean(sparsity_loss).cpu().data[0],
#               'continuity_loss:', torch.mean(continuity_loss).cpu().data[0])
#     print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    

#     eval_sets = ['eval']

#     for set_id, set_name in enumerate(eval_sets):
#         dev_correct = 0.0
#         dev_anti_correct = 0.0
#         dev_cls_correct = 0.0
#         dev_total = 0.0
#         sparsity_total = 0.0
#         dev_count = 0

#         start_id = 0
#         z_total = 0.0
#         z_correct = 0.0
#         z_predict = 0.0

#         pos_sparsity_total = 0.0
#         neg_sparsity_total = 0.0
#         pos_dev_count = 0
#         neg_dev_count = 0

#         num_dev_instance = beer_data.data_sets[set_name].size()

#         for start in range(num_dev_instance / args.batch_size):
#             x_mat, y_vec, z_mat, x_mask = beer_data.get_eval_batch(batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), sort=True)

#             batch_x_ = Variable(torch.from_numpy(x_mat))
#             batch_z_ = Variable(torch.from_numpy(z_mat)).type(torch.FloatTensor)
#             batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
#             batch_y_ = Variable(torch.from_numpy(y_vec))
#             if args.cuda:
#                 batch_x_ = batch_x_.cuda()
#                 batch_z_ = batch_z_.cuda()
#                 batch_m_ = batch_m_.cuda()
#                 batch_y_ = batch_y_.cuda()

#             predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)

#             # calculate classification accuarcy
#             _, y_pred = torch.max(predict, dim=1)
#             _, anti_y_pred = torch.max(anti_predict, dim=1)
#             _, y_cls_pred = torch.max(cls_predict, dim=1)

#             z_correct += torch.sum(batch_z_ * z).cpu().data[0]
#             z_predict += torch.sum(z).cpu().data[0]
#             z_total += torch.sum(batch_z_).cpu().data[0]

#             dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
#             dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
#             dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
#             dev_total += args.batch_size

#             if len(z.shape) == 3:
#                 mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
#                 mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
#                 sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
#                 sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
#             else:
#                 sparsity_ratio = _get_sparsity(z, batch_m_)
#                 sparsity_total += sparsity_ratio.cpu().data.numpy().sum() 
#             dev_count += args.batch_size


#         eval_accs.append(dev_correct / dev_total)

#     print 'dev:', eval_accs[-1], 'sparsity:', sparsity_total / dev_count
#     if z_predict == 0.0:
#         print 'highlighted:', z_predict, 'correct:', z_predict, 'precision:', z_predict, 'recall:', z_predict
#     else:
#         print('highlighted:', z_predict, 'correct:', z_correct, 
#               'precision:', z_correct/z_predict, 'recall:', z_correct/z_total)
        
#     return best_dev_acc


# In[ ]:


def evaluate_rationale_model_glue_for_acl(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, print_train_flag=True, eval_test=False):
    classification_model.eval()
    eval_sets = ['dev']
    if eval_test:
        eval_sets = ['test']
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        continuity_total = 0.0
        sparsity_count_total = 0.0
        continuity_count_total = 0.0
        dev_count = 0

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in range(num_dev_instance / args.batch_size):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)
            cls_predict = predict

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
                sparsity_count_total += sparsity_count.cpu().data.numpy().sum()
                continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
                continuity_total += continuity_ratio.cpu().data.numpy().sum() 
                continuity_count_total += continuity_count.cpu().data.numpy().sum() 
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total

        else:
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total
#             test_accs.append(dev_correct / dev_total)
#             test_anti_accs.append(dev_anti_correct / dev_total)
#             dev_cls_accs.append(dev_cls_correct / dev_total)

#     if (i+1) % display_iteration == 0:
    if print_train_flag:
        print('train:', train_accs[-1], 'consistency_loss:', torch.mean(consistency_loss).cpu().data[0], 
              'sparsity_loss:', torch.mean(sparsity_loss).cpu().data[0],
              'continuity_loss:', torch.mean(continuity_loss).cpu().data[0])
    if not eval_test:
        print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    else:
        print 'test:', dev_accs[-1], 'best test:', best_dev_acc, 'anti test acc:', dev_anti_accs[-1], 'cls test acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count        
    print 'sparsity count:', sparsity_count_total / dev_count, 'continuity count:', continuity_count_total / dev_count
        
    return best_dev_acc


# In[ ]:


def evaluate_rationale_model_glue_for_acl_with_test(classification_model, beer_data, args, dev_accs, test_accs, dev_anti_accs, test_anti_accs, dev_cls_accs, best_dev_acc, eval_accs, print_train_flag=True):
    classification_model.eval()
    eval_sets = ['dev', 'test']
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        continuity_total = 0.0
        sparsity_count_total = 0.0
        continuity_count_total = 0.0
        dev_count = 0

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in range(num_dev_instance / args.batch_size):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)
            cls_predict = predict

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
                sparsity_count_total += sparsity_count.cpu().data.numpy().sum()
                continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
                continuity_total += continuity_ratio.cpu().data.numpy().sum() 
                continuity_count_total += continuity_count.cpu().data.numpy().sum() 
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total

        else:
            test_accs.append(dev_correct / dev_total)
            test_anti_accs.append(dev_anti_correct / dev_total)

#     if (i+1) % display_iteration == 0:
    if print_train_flag:
        print('train:', train_accs[-1], 'consistency_loss:', torch.mean(consistency_loss).cpu().data[0], 
              'sparsity_loss:', torch.mean(sparsity_loss).cpu().data[0],
              'continuity_loss:', torch.mean(continuity_loss).cpu().data[0])
    print 'dev:', dev_accs[-1], 'test:', test_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'anti test acc:', test_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    print 'sparsity count:', sparsity_count_total / dev_count, 'continuity count:', continuity_count_total / dev_count
        
    return best_dev_acc


# In[ ]:


# def output_rationale(batch_x, batch_z, batch_y, idx2word, idx2label, inverse=False, threshold=0.9):
#     outputs = []
#     for (x, z, y) in zip(batch_x, batch_z, batch_y):
#         condition = z >= threshold
#         if inverse:
#             condition = z <= threshold
#         output_tokens = []
#         prev_flag = False
#         for word_index, display_flag in zip(x, condition):
#             word = idx2word[word_index]
#             if display_flag:
#                 if not prev_flag:
#                     output_tokens.append([])
# #                 print len(output_tokens), output_tokens
#                 output_tokens[-1].append(word)
# #                 sys.stdout.write(output_word.encode('ascii', 'ignore'))
#             prev_flag = display_flag
    
#         output_sequences = [' '.join(output_seq) for output_seq in output_tokens]
# #         outputs.append(' '.join(output_tokens) + '\t' + idx2label[y])
#         outputs.append(' <s> '.join(output_sequences) + '\t' + idx2label[y])
#     return outputs

def output_rationale(batch_x, batch_z, batch_y, idx2word, idx2label, inverse=False, threshold=0.9):
    outputs = []
    for (x, z, y) in zip(batch_x, batch_z, batch_y):
        condition = z >= threshold
        if inverse:
            condition = z <= threshold
        output_tokens = []
        for word_index, display_flag in zip(x, condition):
            word = idx2word[word_index]
            if display_flag:
                output_tokens.append(word)
            else:
                output_tokens.append('*')
#         outputs.append(' '.join(output_tokens) + '\t' + idx2label[y])
        outputs.append(' '.join(output_tokens) + '\t' + idx2label[y])
    return outputs

def predict_rationale_model_glue_for_acl(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc):
    classification_model.eval()
    eval_sets = ['dev']
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        continuity_total = 0.0
        sparsity_count_total = 0.0
        continuity_count_total = 0.0
        dev_count = 0
        
        outputs_rationale = [] 
        outputs_non_rationale = []
        outputs_full = []

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in tqdm.tqdm(range(num_dev_instance / args.batch_size)):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)
            cls_predict = predict
    
            outputs_rationale += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=False)
        
            outputs_non_rationale += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=True)
            
            outputs_full += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=False, threshold=0.0)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
                sparsity_count_total += sparsity_count.cpu().data.numpy().sum()
                continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
                continuity_total += continuity_ratio.cpu().data.numpy().sum() 
                continuity_count_total += continuity_count.cpu().data.numpy().sum()  
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total

        else:
            test_accs.append(dev_correct / dev_total)

    print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    print 'sparsity count:', sparsity_count_total / dev_count, 'continuity count:', continuity_count_total / dev_count
        
    return best_dev_acc, outputs_rationale, outputs_non_rationale, outputs_full


# In[ ]:


def evaluate_introspection_model_glue_for_acl(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, print_train_flag=True, eval_test=False):
    classification_model.eval()
    eval_sets = ['dev']
    if eval_test:
        eval_sets = ['test']
        
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        continuity_total = 0.0
        sparsity_count_total = 0.0
        continuity_count_total = 0.0
        dev_count = 0

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in range(num_dev_instance / args.batch_size):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            predict, anti_predict, cls_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
                sparsity_count_total += sparsity_count.cpu().data.numpy().sum()
                continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
                continuity_total += continuity_ratio.cpu().data.numpy().sum() 
                continuity_count_total += continuity_count.cpu().data.numpy().sum() 
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total
            
        else:
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total
#             test_accs.append(dev_correct / dev_total)
#             test_anti_accs.append(dev_anti_correct / dev_total)
#             dev_cls_accs.append(dev_cls_correct / dev_total)

#     if (i+1) % display_iteration == 0:
    if print_train_flag:
        print('train:', train_accs[-1], 'consistency_loss:', torch.mean(consistency_loss).cpu().data[0], 
              'sparsity_loss:', torch.mean(sparsity_loss).cpu().data[0],
              'continuity_loss:', torch.mean(continuity_loss).cpu().data[0])
    if not eval_test:
        print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    else:
        print 'test:', dev_accs[-1], 'best test:', best_dev_acc, 'anti test acc:', dev_anti_accs[-1], 'cls test acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count        
    print 'sparsity count:', sparsity_count_total / dev_count, 'continuity count:', continuity_count_total / dev_count
        
    return best_dev_acc


# In[ ]:


def predict_introspection_model_glue_for_acl(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc):
    classification_model.eval()
    eval_sets = ['dev']
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        continuity_total = 0.0
        sparsity_count_total = 0.0
        continuity_count_total = 0.0
        dev_count = 0
        
        outputs_rationale = [] 
        outputs_non_rationale = []
        outputs_full = []

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in tqdm.tqdm(range(num_dev_instance / args.batch_size)):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            predict, anti_predict, cls_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
#             cls_predict = classification_model.get_C_model_pred(batch_x_, batch_m_)
    
            outputs_rationale += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=False)
        
            outputs_non_rationale += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=True)
            
            outputs_full += output_rationale(batch_x_.cpu().data, z.cpu().data, 
                                                  batch_y_.cpu().data, beer_data.idx_2_word, 
                                                  beer_data.idx2label, inverse=False, threshold=0.0)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
                sparsity_count_total += sparsity_count.cpu().data.numpy().sum()
                continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
                continuity_total += continuity_ratio.cpu().data.numpy().sum() 
                continuity_count_total += continuity_count.cpu().data.numpy().sum()  
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total

        else:
            test_accs.append(dev_correct / dev_total)

    print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count, 'continuity:', continuity_total / dev_count
    print 'sparsity count:', sparsity_count_total / dev_count, 'continuity count:', continuity_count_total / dev_count
        
    return best_dev_acc, outputs_rationale, outputs_non_rationale, outputs_full


# In[ ]:


def evaluate_model(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, dev_rewards, 
                   best_dev_acc, best_dev_reward, eval_accs, print_train_flag=True, introspection_mode=True):
    classification_model.eval()
    eval_sets = ['dev']
    for set_id, set_name in enumerate(eval_sets):
        output = ''
        output2 = ''
        start_id = 8001
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        dev_count = 0
        dev_reward = 0

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in range(num_dev_instance / args.batch_size):
            x_mat, y_vec, x_mask = beer_data.get_batch(set_name, batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), 
                                             sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

            if introspection_mode:
                predict, anti_predict, cls_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
            else:
                predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
                cls_predict = classification_model.forward_cls(batch_x_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)
            
            masked_logits = predict - anti_predict
            _, masked_y_pred = torch.max(masked_logits, dim=1) # (batch_size,)
            masked_prediction = (masked_y_pred == batch_y_).type(torch.cuda.FloatTensor)
            anti_prediction = (anti_y_pred == batch_y_).type(torch.cuda.FloatTensor)
            
            pred_rewards = masked_prediction - anti_prediction
            dev_reward += np.float(pred_rewards.sum().cpu().data[0])

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, _ = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum()
            dev_count += args.batch_size

        if set_name == 'dev':
            dev_accs.append(dev_correct / dev_total)
            dev_anti_accs.append(dev_anti_correct / dev_total)
            dev_cls_accs.append(dev_cls_correct / dev_total)
            
            dev_rewards.append(dev_reward / dev_total)
            
            if dev_correct / dev_total > best_dev_acc and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_acc = dev_correct / dev_total
                
            if dev_reward / dev_total > best_dev_reward and sparsity_total / dev_count < args.highlight_percentage + 0.05:
                best_dev_reward = dev_reward / dev_total

        else:
            test_accs.append(dev_correct / dev_total)

#     if (i+1) % display_iteration == 0:
    if print_train_flag:
        print('train:', train_accs[-1], 'consistency_loss:', torch.mean(consistency_loss).cpu().data[0], 
              'sparsity_loss:', torch.mean(sparsity_loss).cpu().data[0],
              'continuity_loss:', torch.mean(continuity_loss).cpu().data[0])
    print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count
    print 'dev reward:', dev_rewards[-1], 'best dev reward:', best_dev_reward, 'sparsity:', sparsity_total / dev_count

    eval_sets = ['eval']

    for set_id, set_name in enumerate(eval_sets):
        dev_correct = 0.0
        dev_anti_correct = 0.0
        dev_cls_correct = 0.0
        dev_total = 0.0
        sparsity_total = 0.0
        dev_count = 0

        start_id = 0
        z_total = 0.0
        z_correct = 0.0
        z_predict = 0.0

        pos_sparsity_total = 0.0
        neg_sparsity_total = 0.0
        pos_dev_count = 0
        neg_dev_count = 0

        num_dev_instance = beer_data.data_sets[set_name].size()

        for start in range(num_dev_instance / args.batch_size):
            x_mat, y_vec, z_mat, x_mask = beer_data.get_eval_batch(batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size), sort=True)

            batch_x_ = Variable(torch.from_numpy(x_mat))
            batch_z_ = Variable(torch.from_numpy(z_mat)).type(torch.FloatTensor)
            batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
            batch_y_ = Variable(torch.from_numpy(y_vec))
            if args.cuda:
                batch_x_ = batch_x_.cuda()
                batch_z_ = batch_z_.cuda()
                batch_m_ = batch_m_.cuda()
                batch_y_ = batch_y_.cuda()

#             predict, anti_predict, cls_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
            if introspection_mode:
                predict, anti_predict, cls_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
            else:
                predict, anti_predict, z, neg_log_probs = classification_model(batch_x_, batch_m_)
                cls_predict = classification_model.forward_cls(batch_x_, batch_m_)

            # calculate classification accuarcy
            _, y_pred = torch.max(predict, dim=1)
            _, anti_y_pred = torch.max(anti_predict, dim=1)
            _, y_cls_pred = torch.max(cls_predict, dim=1)

            z_correct += torch.sum(batch_z_ * z).cpu().data[0]
            z_predict += torch.sum(z).cpu().data[0]
            z_total += torch.sum(batch_z_).cpu().data[0]

            dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
            dev_anti_correct += np.float((anti_y_pred == batch_y_).sum().cpu().data[0])
            dev_cls_correct += np.float((y_cls_pred == batch_y_).sum().cpu().data[0])
            dev_total += args.batch_size

            if len(z.shape) == 3:
                mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous() #(batch_size, num_label, length)
                mask_expand_ = mask_expand_.view(-1, batch_m_.size(1)) #(batch_size*num_label, length)
                sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() / z.size(0)
            else:
                sparsity_ratio, _ = _get_sparsity(z, batch_m_)
                sparsity_total += sparsity_ratio.cpu().data.numpy().sum() 
            dev_count += args.batch_size


        eval_accs.append(dev_correct / dev_total)

#     print 'dev:', eval_accs[-1], 'sparsity:', sparsity_total / dev_count
#     if z_predict == 0.0:
#         print 'highlighted:', z_predict, 'correct:', z_predict, 'precision:', z_predict
#     else:
#         print 'highlighted:', z_predict, 'correct:', z_correct, 'precision:', z_correct/z_predict
        
    print 'dev:', eval_accs[-1], 'sparsity:', sparsity_total / dev_count
    if z_predict == 0.0:
        print 'highlighted:', z_predict, 'correct:', z_predict, 'precision:', z_predict, 'recall:', z_predict
    else:
        print 'highlighted:', z_predict, 'correct:', z_correct, 'precision:', z_correct/z_predict, 'recall:', z_correct/z_total
        
    return best_dev_acc, best_dev_reward

