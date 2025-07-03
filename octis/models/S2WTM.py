import torch
import torch
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from octis.models.model import AbstractModel
from octis.models.spherical_SWTM.dataset import DocDataset
from octis.models.spherical_SWTM.models.WTM_sp import S2WTM as SphSlicedWTM


class S2WTM(AbstractModel):

    def __init__(self, num_topics=10, num_epochs=100, batch_size=256,
                 use_partitions=False, use_validation=False, num_samples=10,
                 dropout=0.5, learning_rate=1e-3, log_every=1e9, beta=1.0,
                 dist='unif_sphere', loss_type='sph_sw', num_projections=500,
                 ftype='linear', degree=3, p=2
                 ):

        assert not(use_validation and use_partitions), "Validation data is not needed for S2WTM. \
            Please set 'use_validation=False'."
        
        super(S2WTM, self).__init__()
        self.hyperparameters = dict()
        self.hyperparameters['num_topics'] = int(num_topics)
        self.hyperparameters['num_epochs'] = int(num_epochs)
        self.hyperparameters['dropout'] = dropout
        self.hyperparameters['batch_size'] = int(batch_size)
        self.hyperparameters['learning_rate'] = learning_rate
        self.hyperparameters['log_every'] = log_every
        self.hyperparameters['beta'] = beta

        self.hyperparameters['dist'] = dist
        self.hyperparameters['loss_type'] = loss_type

        self.hyperparameters['num_projections'] = num_projections
        self.hyperparameters['p'] = p
        self.hyperparameters['ftype'] = ftype
        self.hyperparameters['degree'] = degree
        
        self.early_stopping = None
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples

    def train_model(self, dataset, hyperparameters=None, top_words=10, save_dir=None):

        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.top_word = top_words
        self.vocab = dataset.get_vocabulary()

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            x_train, x_test, input_size = self.preprocess(train, test)

            self.model = SphSlicedWTM(
                bow_dim=input_size,
                device=self.device,
                taskname=None,
                n_topic=self.hyperparameters['num_topics'],
                dropout=self.hyperparameters['dropout'],
                batch_size=self.hyperparameters['batch_size'],
                learning_rate=self.hyperparameters['learning_rate'],
                num_epochs=self.hyperparameters['num_epochs'],
                log_every=self.hyperparameters['log_every'],
                beta=self.hyperparameters['beta'],
                dist=self.hyperparameters['dist'],
                loss_type=self.hyperparameters['loss_type'],
                num_projections=self.hyperparameters['num_projections'],
                p=self.hyperparameters['p'],
                ftype=self.hyperparameters['ftype'],
                degree=self.hyperparameters['degree']
                )
            
            self.model.train(train_data=x_train,
                             test_data=None,
                             verbose=True,
                             topK=10,
                             )
            
            result = self.get_info()
            result['test-topic-document-matrix'] = self.model.get_doc_topic_distribution(
                x_test,
                n_samples=self.num_samples,
                ).T
        else:
            train = dataset.get_corpus()
            x_train, input_size = self.preprocess(train)
            
            self.model = SphSlicedWTM(
                bow_dim=input_size,
                device=self.device,
                taskname=None,
                n_topic=self.hyperparameters['num_topics'],
                dropout=self.hyperparameters['dropout'],
                batch_size=self.hyperparameters['batch_size'],
                learning_rate=self.hyperparameters['learning_rate'],
                num_epochs=self.hyperparameters['num_epochs'],
                log_every=self.hyperparameters['log_every'],
                beta=self.hyperparameters['beta'],
                dist=self.hyperparameters['dist'],
                loss_type=self.hyperparameters['loss_type'],
                num_projections=self.hyperparameters['num_projections'],
                p=self.hyperparameters['p'],
                ftype=self.hyperparameters['ftype'],
                degree=self.hyperparameters['degree']
                )

            self.model.train(
                train_data=x_train,
                test_data=None,
                verbose=True,
                topK=10)
            
            result = self.get_info()
        
        result['topic-document-matrix'] = self.model.get_doc_topic_distribution(
            x_train,
            n_samples=self.num_samples,
            ).T

        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def get_info(self):
        info = {}
        with torch.no_grad():
            idxes = torch.eye(self.hyperparameters['num_topics']).to(self.device)
            info['topic-word-matrix'] = self.model.get_topic_word_dist(normalize=False)
        info['topics'] = self.model.show_topic_words(topK=self.top_word)
        return info


    def set_default_hyperparameters(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    
    @staticmethod
    def preprocess(train, test=None, validation=None):

        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        dictionary = Dictionary(entire_dataset)
        vocabsize = len(dictionary)

        full_corpus = [dictionary.doc2bow(line) for line in entire_dataset]
        tfidf_model = TfidfModel(full_corpus)

        train_vec = [dictionary.doc2bow(doc) for doc in train]
        tfidf_train_vec = [tfidf_model[vec] for vec in train_vec] 
        train_data = DocDataset(tfidf_train_vec, train, dictionary)

        if test is not None and validation is not None:
            test_vec = [dictionary.doc2bow(doc) for doc in test]
            tfidf_test_vec = [tfidf_model[vec] for vec in test_vec]
            test_data = DocDataset(tfidf_test_vec, test, dictionary)

            valid_vec = [dictionary.doc2bow(doc) for doc in validation]
            tfidf_valid_vec = [tfidf_model[vec] for vec in valid_vec]
            valid_data = DocDataset(tfidf_valid_vec, validation, dictionary)
            return train_data, test_data, valid_data, vocabsize
        
        if test is None and validation is not None:
            valid_vec = [dictionary.doc2bow(doc) for doc in validation]
            tfidf_valid_vec = [tfidf_model[vec] for vec in valid_vec]
            valid_data = DocDataset(tfidf_valid_vec, validation, dictionary)
            return train_data, valid_data, vocabsize
        
        if test is not None and validation is None:
            test_vec = [dictionary.doc2bow(doc) for doc in test]
            tfidf_test_vec = [tfidf_model[vec] for vec in test_vec]
            test_data = DocDataset(tfidf_test_vec, test, dictionary)
            return train_data, test_data, vocabsize
        
        if test is None and validation is None:
            return train_data, vocabsize