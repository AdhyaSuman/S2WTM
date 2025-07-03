from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import multiprocessing as mp
from octis.models.model import AbstractModel
from octis.models.ECRTM_model.runners.Runner import Runner

import os
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch

class ECRTM(AbstractModel):

    def __init__(self, num_topic, learning_rate=0.002,
                 lr_scheduler=True, lr_step_size=125,
                 beta_temp=0.2, en1_units=200, dropout=0,
                 epochs=500, weight_loss_ECR=250,
                 sinkhorn_alpha=20, OT_max_iter=1000,
                 use_partitions=True, use_validation=False,
                 batch_size=200, w2v_path=None, emb_dim=200):
        """
        initialization of ECRTM
        """
        assert not(use_validation and use_partitions), "Validation data is not needed for ECRTM. Please set 'use_validation=False'."

        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters['learning_rate'] = learning_rate
        self.hyperparameters['lr_scheduler'] = lr_scheduler
        self.hyperparameters['lr_step_size'] = lr_step_size
        self.hyperparameters['epochs'] = epochs
        self.hyperparameters['beta_temp'] = beta_temp
        self.hyperparameters['num_topic'] = num_topic
        self.hyperparameters['en1_units'] = en1_units
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['batch_size'] = batch_size
        
        self.hyperparameters['weight_loss_ECR'] = weight_loss_ECR
        self.hyperparameters['sinkhorn_alpha'] = sinkhorn_alpha
        self.hyperparameters['OT_max_iter'] = OT_max_iter

        self.use_partitions = use_partitions
        self.use_validation = use_validation

        self.w2v_path = w2v_path
        self.emb_dim = emb_dim

        self.model = None
        self.vocab = None

    def train_model(self, dataset, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.vocab = dataset.get_vocabulary()
        self.hyperparameters['vocab_size'] = len(self.vocab)

        if not os.path.isdir(self.w2v_path):
            os.makedirs(self.w2v_path)

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            train_loader, train_data,  \
                test_loader, test_data, \
                    word_embeddings = self.preprocess(
                        vocab=self.vocab,
                        train=data_corpus_train,
                        test=data_corpus_test, 
                        batch_size=self.hyperparameters['batch_size'],
                        w2v_path=self.w2v_path+"{}_emb.npz".format(self.emb_dim),
                        emb_dim=self.emb_dim
                        )
            
            args = self.dict2args(self.hyperparameters, word_embeddings)
            runner = Runner(args)

            self.beta = runner.train(train_loader)
            self.train_theta = runner.test(train_data)
            self.test_theta = runner.test(test_data)
            result = self.get_info()
            return result

        else:
            data_corpus = [' '.join(i) for i in dataset.get_corpus()]
            train_loader, train_data, \
                word_embeddings = self.preprocess(
                    self.vocab,
                    train=data_corpus,
                    batch_size=self.hyperparameters['batch_size'],
                    w2v_path=self.w2v_path+"{}_emb.npz".format(self.emb_dim),
                    emb_dim=self.emb_dim
                    )

            args = self.dict2args(self.hyperparameters, word_embeddings)
            runner = Runner(args)

            self.beta = runner.train(train_loader)
            self.train_theta = runner.test(train_data)
            result = self.get_info()
            return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
    
    @staticmethod
    def dict2args(dict, word_embeddings):
        parser = argparse.ArgumentParser(description='Arguments of ECRTM')
        parser.add_argument("-f", required=False)
        for k, v in dict.items():
            parser.add_argument('--' + k, default=v)

        parser.add_argument('--word_embeddings', default=word_embeddings)
        args = parser.parse_args()
        print(args)
        return args
    
    
    def get_info(self, n_top_words=10):
        info = {}
        top_words = list()
        for i in range(len(self.beta)):
            top_words.append([self.vocab[j] for j in self.beta[i].argsort()[-n_top_words:][::-1]])
        info['topic-word-matrix'] = self.beta
        info['topic-document-matrix'] = self.train_theta.T
        if self.use_partitions:
            info['test-topic-document-matrix'] = self.test_theta.T
        info['topics'] = top_words
        return info


    @staticmethod
    def preprocess(vocab, train, test=None, validation=None,
                   batch_size=None, emb_dim=None, w2v_path=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(
            vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}
        x_train = torch.from_numpy(vec.transform(train).todense().astype('float32'))
        
        x_train = x_train.cuda() if torch.cuda.is_available() else x_train

        train_loader = DataLoader(x_train, batch_size=batch_size, shuffle=True)
        input_size = len(idx2token.keys())

        if w2v_path is not None:
            if os.path.exists(w2v_path):
                emb_mat = np.load(w2v_path)
            else:
                emb_mat = ECRTM.w2v_from_list(entire_dataset,
                                              vocab,
                                              save_path=w2v_path,
                                              dim=emb_dim)

        if test is not None and validation is not None:
            x_test = torch.from_numpy(vec.transform(test).todense().astype('float32'))
            x_test = x_test.cuda() if torch.cuda.is_available() else x_test
            test_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)

            x_valid = torch.from_numpy(vec.transform(validation).todense().astype('float32'))
            x_valid = x_valid.cuda() if torch.cuda.is_available() else x_valid
            valid_loader = DataLoader(x_valid, batch_size=batch_size, shuffle=False)

            return train_loader, x_train, test_loader, x_test, valid_loader, x_valid, emb_mat
        
        if test is None and validation is not None:
            x_valid = torch.from_numpy(vec.transform(validation).todense().astype('float32'))
            x_valid = x_valid.cuda() if torch.cuda.is_available() else x_valid
            valid_loader = DataLoader(x_valid, batch_size=batch_size, shuffle=False)
            return train_loader, x_train, valid_loader, x_valid, emb_mat
        
        if test is not None and validation is None:
            x_test = torch.from_numpy(vec.transform(test).todense().astype('float32'))
            x_test = x_test.cuda() if torch.cuda.is_available() else x_test
            test_loader = DataLoader(x_test, batch_size=batch_size, shuffle=False)
            return train_loader, x_train, test_loader, x_test, emb_mat
        
        if test is None and validation is None:
            return train_loader, x_train, emb_mat

    @staticmethod
    def w2v_from_list(texts, vocab, save_path=None,
                    min_count=2, dim=300, epochs=50,
                    workers=mp.cpu_count(), negative_samples=10,
                    window_size=4):
        """
        :param text: list of  documents
        :param vocab: list of words
        """ 
        texts = [s.split() for s in texts]
        model = Word2Vec(texts, min_count=min_count, sg=1, vector_size=dim, epochs=epochs,
                        workers=workers, negative=negative_samples, window=window_size)
        
        embedding_matrix = np.zeros((len(vocab), dim), dtype=np.float64)
        for i,v in enumerate(vocab):
            missing_words=0
            try:
                embedding_matrix[i] = model.wv[v]
            except:
                missing_words += 1
                embedding_matrix[i] = np.random.normal(scale=0.6, size=(dim,))
        
        print('Embeddings are not found for {} words'.format(missing_words))
        #saving the emb matrix
        if save_path:
            np.save(open(save_path, 'wb'), embedding_matrix)
        
        return embedding_matrix