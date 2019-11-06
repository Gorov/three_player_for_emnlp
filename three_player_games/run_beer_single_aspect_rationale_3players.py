
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy, random, sys, os
from collections import deque
# from models.models import CnnModel, RnnModel

# from basic_nlp_models import BasicNLPModel
# from models.encoder import Encoder, ClassificationEncoder
from datasets.beer_dataset_single_aspect import BeerDatasetBinarySingleAspect, BeerDatasetBinarySingleAspectWithTest
# from models.base_classification_models import BasicClassificationModel, SingleHeadAttentionModel
# from models.base_classification_models import BasicAttentionModel, IntraAttentionModel
# from models.rationale_sentence_classification_models import HardRationaleClassificationModel
from rationale_3players_for_emnlp import HardRationale3PlayerClassificationModelForEmnlp
from tqdm import tqdm

from util_functions import copy_classifier_module, evaluate_rationale_model_glue_for_acl

# import matplotlib
# # matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.manual_seed(9527)
np.random.seed(9527)
random.seed(9527)


# In[ ]:


def display_example(dataset, x, z=None, threshold=0.9):
    """
    Given word a suquence of word index, and its corresponding rationale,
    display it
    Inputs:
        x -- input sequence of word indices, (sequence_length,)
        z -- input rationale sequence, (sequence_length,)
        threshold -- display as rationale if z_i >= threshold
    Outputs:
        None
    """
    # apply threshold
    condition = z >= threshold
    for word_index, display_flag in zip(x, condition):
        word = dataset.idx_2_word[word_index]
        if display_flag:
            output_word = "%s %s%s" %(fg(1), word, attr(0))
            sys.stdout.write(output_word.decode('utf-8'))                
        else:
            sys.stdout.write(" " + word.decode('utf-8'))
    sys.stdout.flush()


# In[ ]:


data_dir = "../data/"
beer_data = BeerDatasetBinarySingleAspectWithTest(data_dir, score_threshold=0.6, split_ratio=0.1)

class Argument():
    def __init__(self):
        self.model_type = 'RNN'
        self.cell_type = 'GRU'
        self.hidden_dim = 400
        self.embedding_dim = 100
        self.kernel_size = 5
        self.layer_num = 1
        self.fine_tuning = False
        self.z_dim = 2
        self.gumbel_temprature = 0.1
        self.cuda = True
        self.batch_size = 40
        self.mlp_hidden_dim = 50
        self.dropout_rate = 0.4
        self.use_relative_pos = True
        self.max_pos_num = 20
        self.pos_embedding_dim = -1
        self.fixed_classifier = True
        self.fixed_E_anti = True
        self.lambda_sparsity = 1.0
        self.lambda_continuity = 1.0
        self.lambda_anti = 1.0
        self.lambda_pos_reward = 0.1
        self.exploration_rate = 0.05
        self.highlight_percentage = 0.3
        self.highlight_count = 8
        self.count_tokens = 8
        self.count_pieces = 4
        self.lambda_acc_gap = 1.2
        self.label_embedding_dim = 400
        self.game_mode = '3player'
        self.margin = 0.2
#         self.lm_setting = 'single'
        self.lm_setting = 'multiple'
#         self.lambda_lm = 100.0
        self.lambda_lm = 1.0
        self.ngram = 4
        self.with_lm = False
        self.batch_size_ngram_eval = 5
        self.lr=0.001
        self.working_dir = '/dccstor/yum-dbqa/Rationale/structured_rationale/game_model_with_lm/beer_single_working_dir'
        self.model_prefix = 'tmp.%s.highlight%.2f.cont%.2f'%(self.game_mode, 
                                                                             self.highlight_percentage, 
                                                                             self.lambda_continuity)
        self.pre_trained_model_prefix = 'pre_trained_cls.model'

args = Argument()
args_dict = vars(args)
print vars(args)
# embedding_size = 100

# TODO: handle save/load vocab here, for saving vocab, use the following, for loading, load embedding from checkpoint
embedding_path = "../data/glove.6B.100d.txt"
# embedding_path = '/dccstor/yum-dbqa/pyTorch/ProNet_MTL/sentiment_analysis/glove.840B.300d.txt'
embeddings = beer_data.initial_embedding(args.embedding_dim, embedding_path)
# embeddings = np.load('beer_single_aspect.embedding.npy')

args.num_labels = len(beer_data.label_vocab)
print 'num_labels: ', args.num_labels
print beer_data.idx2label


# In[ ]:

# args.lm_setting = 'single'
# args.game_mode = 'taos'
# args.cuda = False
reload(sys.modules['rationale_3players_for_emnlp'])
reload(sys.modules['util_functions'])
from rationale_3players_for_emnlp import HardRationale3PlayerClassificationModelForEmnlp
from util_functions import copy_classifier_module, evaluate_rationale_model_glue_for_acl

classification_model = HardRationale3PlayerClassificationModelForEmnlp(embeddings, args)

if args.cuda:
    classification_model.cuda()

print classification_model

if 'count_tokens' in args_dict and 'count_pieces' in args_dict:
    classification_model.count_tokens = args.count_tokens
    classification_model.count_pieces = args.count_pieces


# In[ ]:


train_losses = []
train_accs = []
dev_accs = [0.0]
dev_anti_accs = [0.0]
dev_cls_accs = [0.0]
test_accs = [0.0]
best_dev_acc = 0.0

eval_accs = [0.0]
eval_anti_accs = [0.0]

args.load_pre_cls = False
args.load_pre_gen = False

# snapshot_path = os.path.join(args.working_dir, args.pre_trained_model_prefix + '.pt')
# classification_model = torch.load(snapshot_path)

classification_model.init_C_model()

if args.load_pre_cls:
    print('loading pre-trained the CLS')
    snapshot_path_enc = os.path.join(args.working_dir, args.pre_trained_model_prefix + '.encoder.tmp.pt')
    # torch.save(classification_model.generator.Classifier_enc, snapshot_path_enc)
    snapshot_path_pred = os.path.join(args.working_dir, args.pre_trained_model_prefix + '.predictor.tmp.pt')
    # torch.save(classification_model.generator.Classifier_pred, snapshot_path_pred)

    copy_classifier_module(classification_model.E_model, snapshot_path_enc, snapshot_path_pred)
    copy_classifier_module(classification_model.E_anti_model, snapshot_path_enc, snapshot_path_pred)
    
    copy_classifier_module(classification_model.C_model, snapshot_path_enc, snapshot_path_pred)

    print classification_model
if args.load_pre_gen:
    print('loading pre-trained the GEN+CLS')
    snapshot_path_gen = os.path.join(args.working_dir, args.model_prefix + '.train_gen.pt')
    classification_model = torch.load(snapshot_path_gen)
    
#     snapshot_path_gen = '/dccstor/yum-dbqa/Rationale/structured_rationale/relation_extraction/beer_working_dir/rationale_pre_trained_models/rationale.3players.highlight0.10.cls_loss.train_gen.state_dict.bin'
# #     torch.save(classification_model.state_dict(), save_path)
#     check_point = torch.load(snapshot_path_gen)
#     print(check_point.keys())
#     classification_model.load_state_dict(check_point)
#     print classification_model


# In[ ]:


args.pre_train_cls = False

if args.pre_train_cls:
    print('pre-training the classifier')
    train_losses = []
    train_accs = []
    dev_accs = [0.0]
    dev_anti_accs = [0.0]
    dev_cls_accs = [0.0]
    test_accs = [0.0]
    best_dev_acc = 0.0
    num_iteration = 20000
    display_iteration = 400
    test_iteration = 400

    eval_accs = [0.0]
    eval_anti_accs = [0.0]

    classification_model.init_optimizers()

    for i in tqdm(xrange(num_iteration)):
        classification_model.train()

        # sample a batch of data
        x_mat, y_vec, x_mask = beer_data.get_train_batch(batch_size=args.batch_size, sort=True)

        batch_x_ = Variable(torch.from_numpy(x_mat))
        batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
        batch_y_ = Variable(torch.from_numpy(y_vec))
        if args.cuda:
            batch_x_ = batch_x_.cuda()
            batch_m_ = batch_m_.cuda()
            batch_y_ = batch_y_.cuda()
            
        losses, predict = classification_model.train_cls_one_step(batch_x_, batch_y_, batch_m_)

        # calculate classification accuarcy
        _, y_pred = torch.max(predict, dim=1)

        acc = np.float((y_pred == batch_y_).sum().cpu().data[0]) / args.batch_size
        train_accs.append(acc)

        if (i+1) % test_iteration == 0:
            classification_model.eval()
            eval_sets = ['dev']
            for set_id, set_name in enumerate(eval_sets):
                output = ''
                output2 = ''
                dev_correct = 0.0
                dev_anti_correct = 0.0
                dev_cls_correct = 0.0
                dev_total = 0.0
                sparsity_total = 0.0
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

                    predict = classification_model.forward_cls(batch_x_, batch_m_)

                    # calculate classification accuarcy
                    _, y_pred = torch.max(predict, dim=1)

                    dev_correct += np.float((y_pred == batch_y_).sum().cpu().data[0])
                    dev_total += args.batch_size

                    dev_count += args.batch_size

                if set_name == 'dev':
                    dev_accs.append(dev_correct / dev_total)
                    dev_anti_accs.append(dev_anti_correct / dev_total)
                    dev_cls_accs.append(dev_cls_correct / dev_total)
                    if dev_correct / dev_total > best_dev_acc:
                        best_dev_acc = dev_correct / dev_total

                else:
                    test_accs.append(dev_correct / dev_total)

            print('train:', train_accs[-1])
            print 'dev:', dev_accs[-1], 'best dev:', best_dev_acc, 'anti dev acc:', dev_anti_accs[-1], 'cls dev acc:', dev_cls_accs[-1], 'sparsity:', sparsity_total / dev_count
                


# In[ ]:


# args.pre_train_gen = False

# if args.pre_train_gen:
#     print('pre-training the GEN')
#     train_losses = []
#     train_accs = []
#     dev_accs = [0.0]
#     dev_anti_accs = [0.0]
#     dev_cls_accs = [0.0]
#     test_accs = [0.0]
#     best_dev_acc = 0.0
#     num_iteration = 20000
#     display_iteration = 400
#     test_iteration = 400

#     eval_accs = [0.0]
#     eval_anti_accs = [0.0]

#     classification_model.init_optimizers()
#     classification_model.init_rl_optimizers()
#     classification_model.init_reward_queue()

#     old_E_weights = classification_model.E_model.predictor._parameters['weight'][0].cpu().data.numpy()
#     old_E_anti_weights = classification_model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

#     for i in tqdm(xrange(num_iteration)):
#         classification_model.train()

#         # sample a batch of data
#         x_mat, y_vec, x_mask = beer_data.get_train_batch(batch_size=args.batch_size, sort=True)

#         batch_x_ = Variable(torch.from_numpy(x_mat))
#         batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
#         batch_y_ = Variable(torch.from_numpy(y_vec))
#         if args.cuda:
#             batch_x_ = batch_x_.cuda()
#             batch_m_ = batch_m_.cuda()
#             batch_y_ = batch_y_.cuda()

#         losses, predict, anti_predict, z, rewards, continuity_loss, sparsity_loss = classification_model.train_gen_one_step(batch_x_, batch_y_, batch_m_)

#         # calculate classification accuarcy
#         _, y_pred = torch.max(predict, dim=1)

#         acc = np.float((y_pred == batch_y_).sum().cpu().data[0]) / args.batch_size
#         train_accs.append(acc)

#         train_losses.append(losses['g_rl_loss'][0])

#         new_E_weights = classification_model.E_model.predictor._parameters['weight'][0].cpu().data.numpy()
#         new_E_anti_weights = classification_model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

#         assert (old_E_anti_weights == new_E_anti_weights).all(), 'E anti model changed'
#         assert (old_E_weights == new_E_weights).all(), 'E model changed'

#         if (i+1) % display_iteration == 0:
#             print 'sparsity lambda: %.4f'%(classification_model.lambda_sparsity)
#             print 'highlight percentage: %.4f'%(classification_model.highlight_percentage)
#             print 'rl_loss %.4f, ssparsity_loss %.4f, continuity_loss %.4f'%(losses['g_rl_loss'][0], torch.mean(sparsity_loss).cpu().data[0], torch.mean(continuity_loss).cpu().data[0])
#             y_ = y_vec[2]
#             pred_ = y_pred.data[2]
#             x_ = x_mat[2,:]
#             if len(z.shape) == 3:
#                 z_ = z.cpu().data[2,pred_,:]
#             else:
#                 z_ = z.cpu().data[2,:]
                
#             z_b = torch.zeros_like(z)
#             z_b_ = z_b.cpu().data[2,:]
#             print 'gold label:', beer_data.idx2label[y_], 'pred label:', beer_data.idx2label[pred_]
#             beer_data.display_example(x_, z_, z_b_)

#         if (i+1) % test_iteration == 0:
#             new_best_dev_acc = evaluate_rationale_model(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, eval_accs, print_train_flag=False)
            
#             if new_best_dev_acc > best_dev_acc:
#                 best_dev_acc = new_best_dev_acc
                
#                 snapshot_path = os.path.join(args.working_dir, args.model_prefix + '.train_gen.pt')
#                 print('new best dev:', best_dev_acc, 'model saved at', snapshot_path)
#                 torch.save(classification_model, snapshot_path)
                


# In[ ]:


# reload(sys.modules['util_functions'])
# from util_functions import evaluate_rationale_model_glue_for_acl

args.fixed_E_anti = False
classification_model.fixed_E_anti = args.fixed_E_anti
args.with_lm = False
args.lambda_lm = 1.0

print 'training with game mode:', classification_model.game_mode

train_losses = []
train_accs = []
dev_accs = [0.0]
dev_anti_accs = [0.0]
dev_cls_accs = [0.0]
test_accs = [0.0]
test_anti_accs = [0.0]
test_cls_accs = [0.0]
best_dev_acc = 0.0
best_test_acc = 0.0
num_iteration = 40000
display_iteration = 400
test_iteration = 400

eval_accs = [0.0]
eval_anti_accs = [0.0]

queue_length = 200
z_history_rewards = deque(maxlen=queue_length)
z_history_rewards.append(0.)

classification_model.init_optimizers()
classification_model.init_rl_optimizers()
classification_model.init_reward_queue()

old_E_anti_weights = classification_model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

for i in tqdm(xrange(num_iteration)):
    classification_model.train()
#     supervise_optimizer.zero_grad()
#     rl_optimizer.zero_grad()

#     classification_model.lambda_sparsity = (float(i) / num_iteration) * args.lambda_sparsity
#     classification_model.highlight_percentage = args.highlight_percentage + (1.0 - args.highlight_percentage) * (1 - (float(i) / num_iteration))

    # sample a batch of data
    x_mat, y_vec, x_mask = beer_data.get_train_batch(batch_size=args.batch_size, sort=True)

    batch_x_ = Variable(torch.from_numpy(x_mat))
    batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
    batch_y_ = Variable(torch.from_numpy(y_vec))
    if args.cuda:
        batch_x_ = batch_x_.cuda()
        batch_m_ = batch_m_.cuda()
        batch_y_ = batch_y_.cuda()
        
    z_baseline = Variable(torch.FloatTensor([float(np.mean(z_history_rewards))]))
    if args.cuda:
        z_baseline = z_baseline.cuda()
    
    if not args.with_lm:
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = classification_model.train_one_step(
            batch_x_, batch_y_, z_baseline, batch_m_, with_lm=False)
    else:
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = classification_model.train_one_step(
            batch_x_, batch_y_, z_baseline, batch_m_, with_lm=True)
    
    z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
    z_history_rewards.append(z_batch_reward)

    # calculate classification accuarcy
    _, y_pred = torch.max(predict, dim=1)
    
    acc = np.float((y_pred == batch_y_).sum().cpu().data[0]) / args.batch_size
    train_accs.append(acc)

    train_losses.append(losses['e_loss'][0])
    
    if args.fixed_E_anti == True:
        new_E_anti_weights = classification_model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()
        assert (old_E_anti_weights == new_E_anti_weights).all(), 'E anti model changed'
    
    if (i+1) % display_iteration == 0:
        print 'sparsity lambda: %.4f'%(classification_model.lambda_sparsity)
        print 'highlight percentage: %.4f'%(classification_model.highlight_percentage)
        print 'supervised_loss %.4f, sparsity_loss %.4f, continuity_loss %.4f'%(losses['e_loss'][0], torch.mean(sparsity_loss).cpu().data[0], torch.mean(continuity_loss).cpu().data[0])
        if args.with_lm:
            print('lm prob: %.4f'%losses['lm_prob'][0])
#             print('lm prob: %.4f'%np.tanh(losses['lm_prob'][0]))
        y_ = y_vec[2]
        pred_ = y_pred.data[2]
        x_ = x_mat[2,:]
        if len(z.shape) == 3:
            z_ = z.cpu().data[2,pred_,:]
        else:
            z_ = z.cpu().data[2,:]

        z_b = torch.zeros_like(z)
        z_b_ = z_b.cpu().data[2,:]
        print 'gold label:', beer_data.idx2label[y_], 'pred label:', beer_data.idx2label[pred_]
        beer_data.display_example(x_, z_)

    if (i+1) % test_iteration == 0:
        new_best_dev_acc = evaluate_rationale_model_glue_for_acl(classification_model, beer_data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, print_train_flag=False)
        
        new_best_test_acc = evaluate_rationale_model_glue_for_acl(classification_model, beer_data, args, test_accs, test_anti_accs, test_cls_accs, best_test_acc, print_train_flag=False, eval_test=True)

        if new_best_dev_acc > best_dev_acc:
            best_dev_acc = new_best_dev_acc
#             snapshot_path = os.path.join(args.working_dir, args.model_prefix + '.state_dict.bin')
#             print('new best dev:', new_best_dev_acc, 'model saved at', snapshot_path)
#             torch.save(classification_model.state_dict(), snapshot_path)

        if new_best_test_acc > best_test_acc:
            best_test_acc = new_best_test_acc


# In[ ]:




