
# coding: utf-8

# In[ ]:


import os
import sys
import gzip
import random
import numpy as np
from colored import fg, attr, bg
import json

from dataset import SentenceClassification, BeerDatasetBinary, SentenceClassificationSet


# In[ ]:


class BeerDatasetBinarySingleAspect(SentenceClassification):
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
        self.aspect_names = ['apperance', 'aroma', 'palate', 'taste']
        
        super(BeerDatasetBinarySingleAspect, self).__init__(data_dir, truncate_num, freq_threshold)
        
        self.truncated_word_dict = None
        self.truncated_vocab = None
        
        
    def _init_lm_output_vocab(self, truncate_freq=4):
        word_freq_dict = self._get_word_freq({'train':self.data_sets['train']})
        print('size of the raw vocabulary on training: %d'% len(word_freq_dict))
        print('size of the raw vocabulary on training: %d'% len(self.idx_2_word))

        self.truncated_word_dict = {0:0, 1:1, 2:2, 3:3}
        self.truncated_vocab = ['<PAD>', '<START>', '<END>', '<UNK>']

        for wid, word in self.idx_2_word.items():
            if wid < 4:
                continue
            elif word not in word_freq_dict:
                self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]
            else:
                if word_freq_dict[word] >= truncate_freq:
                    self.truncated_word_dict[wid] = len(self.truncated_vocab)
                    self.truncated_vocab.append(word)
                else:
                    self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]

        tmp_dict = {}
        for word, wid in self.truncated_word_dict.items():
            if wid not in tmp_dict:
                tmp_dict[wid] = 1
        print('size of the truncated vocabulary on training: %d'%len(tmp_dict))
        print('size of the truncated vocabulary on training: %d'%len(self.truncated_vocab))
        
        
    def load_dataset(self):
        
        filein = open(os.path.join(self.data_dir, 'sec_name_dict.json'), 'r')
        self.filtered_name_dict = json.load(filein)
        filein.close()
        
        self.data_sets = {}
        
        # load train
        self.data_sets['train'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                   'reviews.aspect{:d}.train.txt.gz'.format(self.aspect)))
        
        # load dev
        self.data_sets['dev'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                 'reviews.aspect{:d}.heldout.txt.gz'.format(self.aspect)))
#         self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 'test.fasttext'))
        
#         self.data_sets['eval'] = self._load_evaluation_set(os.path.join(self.data_dir, 'annotations.json'), 
#                                                            aspect=self.aspect)
    
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
        
        section_name_dict = {}
        
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
                
#                 for i, token in enumerate(tokens):
#                     if token == ':' and i > 0:
#                         name = tokens[i-1]
                        
#                         if name not in section_name_dict:
#                             section_name_dict[name] = 1
#                         else:
#                             section_name_dict[name] += 1
                
                start = -1
                for i, token in enumerate(tokens):
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name == 'a' or name == 'appearance':
                            start = i - 1
                            break
                
                if start < 0:
                    continue
                
                end = -1
                for i, token in enumerate(tokens):
                    if i <= start + 1:
                        continue
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name in self.filtered_name_dict:
                            end = i - 1
                            break
                            
                if end < 0:
                    continue

                data_set.add_one(tokens[start:end], label)
            
        data_set.print_info()
        
#         self.sorted_names = sorted(section_name_dict.items(), key = lambda x: x[1], reverse=True)
#         self.filtered_name_dict = {}
#         for k, v in section_name_dict.items():
#             if v > 400:
#                 self.filtered_name_dict[k] = v
        
#         print self.sorted_names
#         print self.filtered_name_dict

        return data_set


# In[ ]:


import random

class SentenceClassificationSetSubSampling(SentenceClassificationSet):
    '''
    '''
    def __init__(self):
        super(SentenceClassificationSetSubSampling, self).__init__()
    
    def split_datasets(self, ratio):
        data_set_larger = SentenceClassificationSet()
        data_set_smaller = SentenceClassificationSet()
        for instance in self.instances:
            if random.random() > ratio:
                data_set_pointer = data_set_larger
            else:
                data_set_pointer = data_set_smaller
                
            label = instance['label']
                
            data_set_pointer.instances.append(instance)
            if label not in data_set_pointer.label2instance_dict:
                data_set_pointer.label2instance_dict[label] = {}

            data_set_pointer.label2instance_dict[label][len(data_set_pointer.instances)] = 1
            
        return data_set_larger, data_set_smaller


# In[ ]:


class BeerDatasetBinarySingleAspectWithTest(SentenceClassification):
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1, aspect=0, score_threshold=0.5, split_ratio=0.15):
        """
        This function initialize a data set from Beer Review:
        Inputs:
            data_dir -- the directory containing the data
            aspect -- an integer of an aspect from 0-4
            truncate_num -- max length of the review text to use
        """
        self.aspect = aspect
        self.score_threshold = score_threshold
        self.aspect_names = ['apperance', 'aroma', 'palate', 'taste']
        self.split_ratio = split_ratio
        
        super(BeerDatasetBinarySingleAspectWithTest, self).__init__(data_dir, truncate_num, freq_threshold)
        
        self.truncated_word_dict = None
        self.truncated_vocab = None
        
        
    def _init_lm_output_vocab(self, truncate_freq=4):
        word_freq_dict = self._get_word_freq({'train':self.data_sets['train']})
        print('size of the raw vocabulary on training: %d'% len(word_freq_dict))
        print('size of the raw vocabulary on training: %d'% len(self.idx_2_word))

        self.truncated_word_dict = {0:0, 1:1, 2:2, 3:3}
        self.truncated_vocab = ['<PAD>', '<START>', '<END>', '<UNK>']

        for wid, word in self.idx_2_word.items():
            if wid < 4:
                continue
            elif word not in word_freq_dict:
                self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]
            else:
                if word_freq_dict[word] >= truncate_freq:
                    self.truncated_word_dict[wid] = len(self.truncated_vocab)
                    self.truncated_vocab.append(word)
                else:
                    self.truncated_word_dict[wid] = self.truncated_word_dict[self.word_vocab['<UNK>']]

        tmp_dict = {}
        for word, wid in self.truncated_word_dict.items():
            if wid not in tmp_dict:
                tmp_dict[wid] = 1
        print('size of the truncated vocabulary on training: %d'%len(tmp_dict))
        print('size of the truncated vocabulary on training: %d'%len(self.truncated_vocab))
        
        
    def load_dataset(self):
        
        filein = open(os.path.join(self.data_dir, 'sec_name_dict.json'), 'r')
        self.filtered_name_dict = json.load(filein)
        filein.close()
        
        self.data_sets = {}
        
        # load train
        tmp_dataset = self._load_data_set(os.path.join(self.data_dir, 
                                                       'reviews.aspect{:d}.train.txt.gz'.format(self.aspect)),
                                         with_dev=True)
        
        print('splitting with %.2f'%self.split_ratio)
        self.data_sets['train'], self.data_sets['dev'] = tmp_dataset.split_datasets(self.split_ratio)
        self.data_sets['train'].print_info()
        self.data_sets['dev'].print_info()
        
        # load dev
        self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                 'reviews.aspect{:d}.heldout.txt.gz'.format(self.aspect)))
    
        # build vocab
        self._build_vocab()
        
        self.idx2label = {val: key for key, val in self.label_vocab.items()}
        
        
    def _load_data_set(self, fpath, with_dev=False):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        if with_dev:
            data_set = SentenceClassificationSetSubSampling()
        else:
            data_set = SentenceClassificationSet()
        
        section_name_dict = {}
        
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
                
                start = -1
                for i, token in enumerate(tokens):
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name == 'a' or name == 'appearance':
                            start = i - 1
                            break
                
                if start < 0:
                    continue
                
                end = -1
                for i, token in enumerate(tokens):
                    if i <= start + 1:
                        continue
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name in self.filtered_name_dict:
                            end = i - 1
                            break
                            
                if end < 0:
                    continue

                data_set.add_one(tokens[start:end], label)
            
        data_set.print_info()

        return data_set


# In[ ]:


if __name__ == "__main__":
    test_case = 'beer'
    
    if test_case == 'beer': 
        data_dir = "/dccstor/yum-dbqa/Rationale/deep-rationalization/beer_review/"

        beer_data = BeerDatasetBinarySingleAspect(data_dir, score_threshold=0.6)

        x_mat, y_vec, x_mask = beer_data.get_batch('dev', range(2), sort=False)
#         x_mat, y_vec, z_mat, x_mask = beer_data.get_eval_batch(range(2), sort=False)
        print y_vec
#         print x_mask
        print x_mat[1]
#         print z_mat[1]

        print beer_data.label_vocab
        beer_data.display_sentence(x_mat[0])
        beer_data.display_sentence(x_mat[1])
        
#         beer_data.display_example(x_mat[0], z_mat[0])
#         beer_data.display_example(x_mat[1], z_mat[1])
        


# In[ ]:


# beer_data = BeerDatasetBinarySingleAspectWithTest(data_dir, score_threshold=0.6)


# In[ ]:


# fileout = open(os.path.join(data_dir, 'sec_name_dict.json'), 'w')
# json.dump(beer_data.filtered_name_dict, fileout)
# fileout.close()


# In[ ]:


# print beer_data.filtered_name_dict


# In[ ]:


# print(8511 + 4975)
# print(6629 + 6733)


# In[ ]:




