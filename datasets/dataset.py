
# coding: utf-8

# In[ ]:


import os
import sys
import gzip
import random
import numpy as np
from colored import fg, attr, bg
import json


# In[ ]:


class TextDataset(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_dataset(self):
        pass
    
    def initial_embedding(self, embedding_size, embedding_path=None):
        """
        This function initialize embedding with glove embedding. 
        If a word has embedding in glove, use the glove one.
        If not, initial with random.
        Inputs:
            embedding_size -- the dimension of the word embedding
            embedding_path -- the path to the glove embedding file
        Outputs:
            embeddings -- a numpy matrix in shape of (vocab_size, embedding_dim)
                          the ith row indicates the word with index i from word_ind_dict
        """    
        vocab_size = len(self.word_vocab)
        # initialize a numpy embedding matrix 
        embeddings = 0.1*np.random.randn(vocab_size, embedding_size).astype(np.float32)
        # replace <PAD> by all zero
        embeddings[0, :] = np.zeros(embedding_size, dtype=np.float32)

        if embedding_path and os.path.isfile(embedding_path):
            f = open(embedding_path, "r")
            counter = 0
            for line in f:
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = data[1::]
                embedding = map(np.float32, embedding)
                if word in self.word_vocab:
                    embeddings[self.word_vocab[word], :] = embedding
                    counter += 1
            f.close()
            print "%d words has been switched." %counter            
        else:
            print "embedding is initialized fully randomly."
            
        return embeddings
    
    def data_to_index(self):
        pass
    
    def _index_to_word(self):
        """
        Apply reverse operation of word to index.
        """
        return {value: key for key, value in self.word_vocab.items()}
    
    def get_batch(self, dataset_id, batch_idx):
        """
        randomly select a batch from a dataset
        
        """
        pass
    
    
    def get_random_batch(self, dataset_id, batch_size):
        """
        randomly select a batch from the training set
        Inputs:
            dataset_id: a integer index of the dataset to sample, 0: train, 1: validation, 2: test
            batch_size: integer
        """
        pass
    
    
    def display_example(self, x, z=None, threshold=0.9):
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
            word = self.idx_2_word[word_index]
            if display_flag:
                output_word = "%s %s%s" %(fg(1), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))                
            else:
                sys.stdout.write(" " + word.encode('utf-8'))
        sys.stdout.flush()


# In[ ]:


class SentenceClassificationSet(object):
    '''
    '''
    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, tokens, label):
        self.instances.append({'sentence':' '.join(tokens), 'label':label})
        if label not in self.label2instance_dict:
            self.label2instance_dict[label] = {}
        
        self.label2instance_dict[label][len(self.instances)] = 1
        
    def get_pairs(self):
        return self.instances
    
    def size(self):
        return len(self.instances)
    
    def get_samples_from_one_list(self, batch_idx, truncate_num=0):
        xs = []
        ys = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            
            sentence = pair_dict_['sentence']
            
            if truncate_num > 0:
                sentence = sentence[:truncate_num]
            if len(sentence) > max_x_len:
                max_x_len = len(sentence)
                
            xs.append(sentence)
            
        return xs, ys, max_x_len
    
    
    def get_ngram_samples_from_one_list(self, batch_idx, ngram=3, truncated_map=None):
        xs = []
        ys = []

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            sentence = pair_dict_['sentence']

            for start_idx in range(len(sentence) - (ngram - 1)):
                context_idxs = sentence[start_idx:start_idx+ngram-1]

                target_idx = sentence[start_idx+ngram-1]
#                 if truncated_map is not None:
#                     target_idx = truncated_map[target_idx]

                xs.append(context_idxs)
                ys.append(target_idx)

        return xs, ys
    
    def get_labeled_ngram_samples_from_one_list(self, trg_label, batch_idx, ngram=3, truncated_map=None):
        xs = []
        ys = []

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            
            if label != trg_label:
                continue
            
            sentence = pair_dict_['sentence']

            for start_idx in range(len(sentence) - (ngram - 1)):
                context_idxs = sentence[start_idx:start_idx+ngram-1]

                target_idx = sentence[start_idx+ngram-1]
#                 if truncated_map is not None:
#                     target_idx = truncated_map[target_idx]

                xs.append(context_idxs)
                ys.append(target_idx)

        return xs, ys
    
    
    def get_eval_samples_from_one_list(self, batch_idx, truncate_num=0):
        xs = []
        zs = []
        ys = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            
            sentence = pair_dict_['sentence']
            highlights = pair_dict_['z']
            
            if truncate_num > 0:
                sentence = sentence[:truncate_num]
                highlights = highlights[:truncate_num]
            if len(sentence) > max_x_len:
                max_x_len = len(sentence)
                
            xs.append(sentence)
            zs.append(highlights)
            
        return xs, ys, zs, max_x_len
    
            
    def print_info(self):
        for k, v in self.label2instance_dict.items():
            print 'Number of instances with label%d:'%k, len(v)


class SentenceClassification(TextDataset):
    
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1):        
        super(SentenceClassification, self).__init__(data_dir)
        self.truncate_num = truncate_num
        self.freq_threshold = freq_threshold
        
        self.word_vocab = {'<PAD>':0, '<START>':1, '<END>':2, '<UNK>':3}
        self.label_vocab = {}
        self.load_dataset()
        print 'Converting text to word indicies.'
        self.idx_2_word = self._index_to_word()
        
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        # load train
        self.data_sets['train'] = self._load_data_set(os.path.join(self.data_dir, 'train.fasttext'))
        
        # load dev
        self.data_sets['dev'] = self._load_data_set(os.path.join(self.data_dir, 'dev.fasttext'))
        self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 'test.fasttext'))
        
        # build vocab
        self._build_vocab()
        
        
    def _load_data_set(self, fpath):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        data_set = SentenceClassificationSet()
        
        f = open(fpath, "r")
        instances = f.readlines()
        
        for idx, instance in enumerate(instances):
            tokens = instance.strip('\n').split(' ')
            label = tokens[0]
            tokens = tokens[1:]
            if label not in self.label_vocab:
                self.label_vocab[label] = len(self.label_vocab)
                label = self.label_vocab[label]
            else:
                label = self.label_vocab[label]
            
            data_set.add_one(tokens, label)
            
        data_set.print_info()

        return data_set
    
    def _build_vocab(self):
        """
        Filter the vocabulary and numeralization
        """
        
        def _add_vocab_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(' ')
            word_idx_list = []
            for token in tokens:
                if word_freq_dict[token] < self.freq_threshold:
                    word_idx_list.append(self.word_vocab['<UNK>'])
                else:
                    if token not in self.word_vocab:
                        self.word_vocab[token] = len(self.word_vocab)
                    word_idx_list.append(self.word_vocab[token])
            return word_idx_list
        
        # numeralize passages in training pair lists
        def _numeralize_pairs(word_freq_dict, pairs):
            ret_pair_list = []
            for pair_dict_ in pairs:
                new_pair_dict_ = {}
                
                for k, v in pair_dict_.items():
                    if k == 'sentence':
                        new_pair_dict_[k] = _add_vocab_from_sentence(word_freq_dict, v)
                    else:
                        new_pair_dict_[k] = pair_dict_[k] 
                
                ret_pair_list.append(new_pair_dict_)
            return ret_pair_list
        
        
        word_freq_dict = self._get_word_freq(self.data_sets)
            
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = _numeralize_pairs(word_freq_dict, data_set.get_pairs())

        print 'size of the final vocabulary:', len(self.word_vocab)
        
        
    def _get_word_freq(self, data_sets_):
        """
        Building word frequency dictionary and filter the vocabulary
        """
        
        def _add_freq_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(' ')
            for token in tokens:
                if token not in word_freq_dict:
                    word_freq_dict[token] = 1
                else:
                    word_freq_dict[token] += 1

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                sentence = pair_dict['sentence']
                _add_freq_from_sentence(word_freq_dict, sentence)

        print 'size of the raw vocabulary:', len(word_freq_dict)
        return word_freq_dict
    
    
    def get_train_batch(self, batch_size, sort=False):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_size: 
        Outputs:
            q_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            p_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
        """
        
        set_id = 'train'
        data_set = self.data_sets[set_id]
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        
        return self.get_batch(set_id, batch_idx, sort)
    
    def get_batch(self, set_id, batch_idx, sort=False):
        
        data_set = self.data_sets[set_id]
        xs_, ys_, max_x_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x_masks_ = []
        for i, x in enumerate(xs_):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            x_masks_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))
            
        x_mat = np.array(xs_, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        y_vec = np.array(ys_, dtype=np.int64)
        
        if sort:
            # sort all according to q_length
            x_length = np.sum(x_mask, axis=1)
            x_sort_idx = np.argsort(-x_length)
            x_mat = x_mat[x_sort_idx, :]
            x_mask = x_mask[x_sort_idx, :]
            y_vec = y_vec[x_sort_idx]
        
        return x_mat, y_vec, x_mask

    
    def display_sentence(self, x):
        """
        Display a suquence of word index
        Inputs:
            x -- input sequence of word indices, (sequence_length,)
        Outputs:
            None
        """
        # apply threshold
        for word_index in x:
            word = self.idx_2_word[word_index]
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()
        


# In[ ]:


class BeerDatasetBinary(SentenceClassification):
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1, aspect=0, score_threshold=0.5):
        """
        This function initialize a data set from Beer Review:
        Inputs:
            data_dir -- the directory containing the data
            aspect -- an integer of an aspect from 0-4
            truncate_num -- max length of the review text to use
        """
        self.aspect = aspect
        self.score_threshold = score_threshold
        super(BeerDatasetBinary, self).__init__(data_dir, truncate_num, freq_threshold)
        
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        # load train
        self.data_sets['train'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                   'reviews.aspect{:d}.train.txt.gz'.format(self.aspect)))
        
        # load dev
        self.data_sets['dev'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                 'reviews.aspect{:d}.heldout.txt.gz'.format(self.aspect)))
#         self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 'test.fasttext'))
        
        self.data_sets['eval'] = self._load_evaluation_set(os.path.join(self.data_dir, 'annotations.json'), 
                                                           aspect=self.aspect)
    
        # build vocab
        self._build_vocab()
        
        self.idx2label = {val: key for key, val in self.label_vocab.items()}
        
        
    def _load_data_set(self, fpath):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        data_set = SentenceClassificationSet()
        
        with gzip.open(os.path.join(fpath), 'r') as f:
            for idx, line in enumerate(f):
                lbl, txt = tuple(line.decode('utf-8').strip('\n').split('\t'))
                lbl = float(lbl.split(' ')[self.aspect])
                
                if lbl > self.score_threshold:
                    label = 'positive'
                else:
                    label = 'negative'
                    
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                    label = self.label_vocab[label]
                else:
                    label = self.label_vocab[label]
                
                txt = txt.split()[:self.truncate_num]
                tokens = [term.lower() for term in txt if term != '']

                data_set.add_one(tokens, label)
            
        data_set.print_info()

        return data_set
    
    
    def _load_evaluation_set(self, fpath, aspect=0):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        print 'loading evaluation set: %s'%fpath
        
        data_set = SentenceClassificationSet()
        
        with open(fpath, 'r') as f:
            for idx, line in enumerate(f):
                inst_data = json.loads(line)
                rationale_lbl = inst_data[str(aspect)] # list of pairs
                lbl = inst_data['y'][aspect] # float
                txt = inst_data['x'] # list of tokens
                
                if lbl > self.score_threshold:
                    label = 'positive'
                else:
                    label = 'negative'
                    
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                    label = self.label_vocab[label]
                else:
                    label = self.label_vocab[label]
                
#                 txt = txt.split()[:self.truncate_num]
                tokens = [term.lower() for term in txt if term != '']
                assert len(tokens) == len(txt)

                data_set.add_one(tokens, label)
                z = [0.0] * len(tokens)
                
                for pair in rationale_lbl:
                    start = pair[0]
                    end = pair[1]
                    for j in range(start, end):
                        z[j] = 1.0
                
                data_set.instances[idx]['z'] = z
            
        data_set.print_info()

        return data_set
    
    
    def get_eval_batch(self, batch_idx, sort=False):
        data_set = self.data_sets['eval']
        xs_, ys_, zs_, max_x_len_ = data_set.get_eval_samples_from_one_list(batch_idx)

        x_masks_ = []
        for i, (x, z) in enumerate(zip(xs_, zs_)):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            zs_[i] = z + (max_x_len_ - len(z)) * [0]
            x_masks_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))
            
        x_mat = np.array(xs_, dtype=np.int64)
        z_mat = np.array(zs_, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        y_vec = np.array(ys_, dtype=np.int64)
        
        if sort:
            # sort all according to q_length
            x_length = np.sum(x_mask, axis=1)
            x_sort_idx = np.argsort(-x_length)
            x_mat = x_mat[x_sort_idx, :]
            z_mat = z_mat[x_sort_idx, :]
            x_mask = x_mask[x_sort_idx, :]
            y_vec = y_vec[x_sort_idx]
        
        return x_mat, y_vec, z_mat, x_mask
    
    
#     def display_example(self, x, z=None, threshold=0.9):
#         """
#         Given word a suquence of word index, and its corresponding rationale,
#         display it
#         Inputs:
#             x -- input sequence of word indices, (sequence_length,)
#             z -- input rationale sequence, (sequence_length,)
#             threshold -- display as rationale if z_i >= threshold
#         Outputs:
#             None
#         """
#         # apply threshold
#         condition = z >= threshold
#         for word_index, display_flag in zip(x, condition):
#             word = self.idx_2_word[word_index]
#             if display_flag:
#                 output_word = "%s %s%s" %(fg(1), word, attr(0))
#                 sys.stdout.write(output_word.encode('utf-8'))                
#             else:
#                 sys.stdout.write(" " + word.encode('utf-8'))
#         sys.stdout.write("\n")
#         sys.stdout.flush()
        
    def display_example(self, x, z_a=None, z_b=None, threshold=0.9):
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
        condition_a = z_a >= threshold
        condition_b = z_b >= threshold
        for word_index, display_flag_a, display_flag_b in zip(x, condition_a, condition_b):
            word = self.idx_2_word[word_index]
            if display_flag_a:
                output_word = "%s %s%s" %(fg(1), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))                
            elif display_flag_b:
                output_word = "%s %s%s" %(fg(2), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))                
            else:
                sys.stdout.write(" " + word.encode('utf-8'))
        sys.stdout.write("\n")
        sys.stdout.flush()
        


# In[ ]:


class RelationClassificationSet(SentenceClassificationSet):
    '''
    '''
    def __init__(self):
        super(RelationClassificationSet, self).__init__()
        
    def add_one(self, tokens, label, ent1_s, ent1_e, ent2_s, ent2_e):
        self.instances.append({'sentence':' '.join(tokens), 'label':label, 
                               'ent1_s':ent1_s, 'ent1_e':ent1_e, 
                               'ent2_s':ent2_s, 'ent2_e':ent2_e})
        if label not in self.label2instance_dict:
            self.label2instance_dict[label] = {}
        
        self.label2instance_dict[label][len(self.instances)] = 1
    
    
    def get_samples_from_one_list(self, batch_idx, truncate_num=0):
        xs = []
        ys = []
        ent1_s = []
        ent1_e = []
        ent2_s = []
        ent2_e = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            ent1_s.append(pair_dict_['ent1_s'])
            ent1_e.append(pair_dict_['ent1_e'])
            ent2_s.append(pair_dict_['ent2_s'])
            ent2_e.append(pair_dict_['ent2_e'])
            
            sentence = pair_dict_['sentence']
            
            if truncate_num > 0:
                sentence = sentence[:truncate_num]
            if len(sentence) > max_x_len:
                max_x_len = len(sentence)
                
            xs.append(sentence)
            
        return xs, ys, ent1_s, ent1_e, ent2_s, ent2_e, max_x_len


class RelationClassification(SentenceClassification):
    
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1):        
        super(RelationClassification, self).__init__(data_dir, truncate_num, freq_threshold)
        
        
    def load_dataset(self):
        
        self.data_sets = {}
        
        # load train
        self.data_sets['train'] = self._load_data_set(os.path.join(self.data_dir, 'SemEval.train.tok'))
        
        # load dev
        self.data_sets['dev'] = self._load_data_set(os.path.join(self.data_dir, 'SemEval.test.tok'))
        
        # build vocab
        self._build_vocab()
        
        
    def _load_data_set(self, fpath):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        data_set = RelationClassificationSet()
        
        f = open(fpath, "r")
        instances = f.read().strip('\n\n').split('\n\n')
        
        for idx in range(len(instances)):
            lines = instances[idx].split('\n')
            head = lines[0].split('\t')
            tokens = lines[1].lower().split(' ')

            label = head[0]
            ent1_s = int(head[1])
            ent1_e = int(head[2])
            ent2_s = int(head[4])
            ent2_e = int(head[5])
            
            if label not in self.label_vocab:
                self.label_vocab[label] = len(self.label_vocab)
                label = self.label_vocab[label]
            else:
                label = self.label_vocab[label]
            
            data_set.add_one(tokens, label, ent1_s, ent1_e, ent2_s, ent2_e)
            
        data_set.print_info()
        
        self.idx2label = {val: key for key, val in self.label_vocab.items()}

        return data_set
    
    
    def get_train_batch(self, batch_size, sort=False):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_size: print
        Outputs:
            q_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            p_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
        """
        
        set_id = 'train'
        data_set = self.data_sets[set_id]
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        
        return self.get_batch(set_id, batch_idx, sort)
    
    def get_batch(self, set_id, batch_idx, sort=False, max_pos_num=5):
        
        data_set = self.data_sets[set_id]
        xs_, ys_, ent1_s, ent1_e, ent2_s, ent2_e, max_x_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x_masks_ = []
        es_ = []
        pos_e1_ = []
        pos_e2_ = []
        for i, x in enumerate(xs_):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            x_masks_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))
            
            es_.append([0] * max_x_len_)
            pos_e1_.append([0] * max_x_len_)
            pos_e2_.append([0] * max_x_len_)
            for j in range(ent1_s[i], ent1_e[i] + 1):
                es_[i][j] = 1
            for j in range(ent2_s[i], ent2_e[i] + 1):
                es_[i][j] = 2

            for j in range(max_x_len_):
                if j < ent1_s[i]:
                    pos_e1_[i][j] = j - ent1_s[i]
                    if pos_e1_[i][j] <= - max_pos_num:
                        pos_e1_[i][j] = - max_pos_num
                elif j > ent1_e[i]:
                    pos_e1_[i][j] = j - ent1_e[i]
                    if pos_e1_[i][j] >= max_pos_num:
                        pos_e1_[i][j] = max_pos_num
                if j < ent2_s[i]:
                    pos_e2_[i][j] = j - ent2_s[i]
                    if pos_e2_[i][j] <= - max_pos_num:
                        pos_e2_[i][j] = - max_pos_num
                elif j > ent2_e[i]:
                    pos_e2_[i][j] = j - ent2_e[i]
                    if pos_e2_[i][j] >= max_pos_num:
                        pos_e2_[i][j] = max_pos_num
                    
        x_mat = np.array(xs_, dtype=np.int64)
        e_mat = np.array(es_, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        y_vec = np.array(ys_, dtype=np.int64)
        pos_e1_mat = np.array(pos_e1_, dtype=np.int64)
        pos_e2_mat = np.array(pos_e2_, dtype=np.int64)
        
#         print pos_e1_mat
#         print pos_e2_mat
        pos_e1_mat = pos_e1_mat + max_pos_num
        pos_e2_mat = pos_e2_mat + max_pos_num
#         print pos_e1_mat
#         print pos_e2_mat
        
        if sort:
            # sort all according to q_length
            x_length = np.sum(x_mask, axis=1)
            x_sort_idx = np.argsort(-x_length)
            x_mat = x_mat[x_sort_idx, :]
            e_mat = e_mat[x_sort_idx, :]
            x_mask = x_mask[x_sort_idx, :]
            y_vec = y_vec[x_sort_idx]
            pos_e1_mat = pos_e1_mat[x_sort_idx, :]
            pos_e2_mat = pos_e2_mat[x_sort_idx, :]
        
        return x_mat, y_vec, e_mat, x_mask, pos_e1_mat, pos_e2_mat

    
    def display_example(self, x, z=None, e=None, threshold=0.9):
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
        for word_index, e_state, display_flag in zip(x, e, condition):
            word = self.idx_2_word[word_index]
            if e_state == 1 and display_flag:
                output_word = " %s%s%s%s" %(bg(6), fg(1), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))
            elif e_state == 2 and display_flag:
                output_word = " %s%s%s%s" %(bg(7), fg(1), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))
            elif e_state == 1:
                output_word = " %s%s%s" %(bg(6), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))
            elif e_state == 2:
                output_word = " %s%s%s" %(bg(7), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))
            elif display_flag:
                output_word = " %s%s%s" %(fg(1), word, attr(0))
                sys.stdout.write(output_word.encode('utf-8'))                
            else:
                sys.stdout.write(" " + word.encode('utf-8'))
        sys.stdout.flush()
        
        
    def display_rationales(self, x, z=None, e=None, threshold=0.9, without_args=False):
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
        output = ''
        for word_index, e_state, display_flag in zip(x, e, condition):
            word = self.idx_2_word[word_index]
            if display_flag:
                if without_args:
                    if e_state != 0:
                        continue
                else:
                    if e_state == 1:
                        word = '<e1>'
                    elif e_state == 2:
                        word = '<e2>'
                output += word + ' '
                        
#         sys.stdout.write('\n')
#         sys.stdout.flush()
        output.strip(' ')
        return output
#         return output.encode('utf-8')


# In[ ]:


def _test_ask_ubuntu():
    data_dir = "/home/chang/storage/askubuntu"
    askubuntu_data = AskUbuntuDataset(data_dir)
    
    embedding_size = 200
    embedding_path = data_dir + "/vector/vectors_pruned.200.txt"
    embeddings = askubuntu_data.initial_embedding(embedding_size, embedding_path)


# In[ ]:


def _test_init_embedding():
    """
    A test function, to check embedding init for beer rewivew dataset.
    """
    data_dir = "/home/chang/storage/beer_review/"
    beer_data = BeerDataset(data_dir, aspect=0)
    word_ind_dict = beer_data.word_vocab
    embedding_size = 50
    embedding_path = "/home/chang/storage/glove/glove.6B.50d.txt"
    embeddings = beer_data.initial_embedding(embedding_size, embedding_path)


# In[ ]:


def _test_numerization():
    data_dir = "/home/chang/storage/beer_review/"
    beer_data = BeerDataset(data_dir, aspect=0)
    data = beer_data.dataset_list[1]
    batch_size = 16
    x, y = beer_data.get_random_batch(1, batch_size)


# In[ ]:


if __name__ == "__main__":
#     _test_ask_ubuntu()

#     data_dir = "/Users/yum/Documents/IBM work/QA_eviagg/MedNLI"

#     mednli_data = MedNLI(data_dir)
    
#     q_mat, p_mat, y_vec, q_mask, p_mask  = mednli_data.get_batch('dev', [1], sort=False)
#     print y_vec
#     print q_mat[0]
#     mednli_data.display_sentence(q_mat[0])
#     print ''
#     mednli_data.display_sentence(p_mat[0])
    test_case = 'beer'
    
    if test_case == 'semeval': 
        data_dir = "/dccstor/yum-dbqa/Rationale/structured_rationale/FCM_nips_workshop/data/"

        disease_data = RelationClassification(data_dir)

        x_mat, y_vec, e_mat, x_mask, pos_e1_mat, pos_e2_mat  = disease_data.get_batch('dev', range(2), sort=False)
        print y_vec
        print x_mask
        print x_mat[1]
        print 'e_mat', e_mat
        print disease_data.label_vocab
        disease_data.display_sentence(x_mat[0])
        disease_data.display_sentence(x_mat[1])

        print x_mat
        print x_mask

        print e_mat
        print pos_e1_mat
        print pos_e2_mat
        print pos_e1_mat - 5
        print pos_e2_mat - 5
    elif test_case == 'beer': 
        data_dir = "/dccstor/yum-dbqa/Rationale/deep-rationalization/beer_review/"

        beer_data = BeerDatasetBinary(data_dir, score_threshold=0.6)

#         x_mat, y_vec, x_mask = beer_data.get_batch('dev', range(2), sort=False)
        x_mat, y_vec, z_mat, x_mask = beer_data.get_eval_batch(range(2), sort=False)
        print y_vec
#         print x_mask
        print x_mat[1]
        print z_mat[1]

        print beer_data.label_vocab
        beer_data.display_sentence(x_mat[0])
        beer_data.display_sentence(x_mat[1])
        
        beer_data.display_example(x_mat[0], z_mat[0])
        beer_data.display_example(x_mat[1], z_mat[1])
        


# In[ ]:




