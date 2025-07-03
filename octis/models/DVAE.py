from octis.models.model import AbstractModel
from octis.models.ntms.models.dvae_rsvi import DVAE_RSVI as DVAE_RSVITM
from octis.models.ntms.models.dvae import DVAE as DVAETM
from octis.models.ntms.models.prod_lda import ProdLDA as DVAE_ProdLDA
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer

class DVAE(AbstractModel):

    def __init__(self, model_name='DVAE_RSVITM', num_topics=10, num_epochs=100, batch_size=128, top_word=10,
                 use_partitions=False, use_validation=False, 
                 ):
        
        assert model_name in ['DVAE_RSVITM', 'DVAETM', 'DVAE_ProdLDA'], f"Unexpected model name: {model_name}"
        
        super(DVAE, self).__init__()
        self.hyperparameters = dict()
        self.model_name = model_name
        self.hyperparameters['num_topics'] = int(num_topics)
        self.hyperparameters['num_epochs'] = int(num_epochs)
        self.hyperparameters['batch_size'] = int(batch_size)
        
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.model = None
        self.top_word = top_word

    def train_model(self, dataset, hyperparameters=None, top_words=10):

        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.top_word = top_words
        vocab = dataset.get_vocabulary()
        
        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            
            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]
            
            train_loader, test_loader, self.id2word = self.preprocess(vocab, data_corpus_train, data_corpus_test, batch_size=self.hyperparameters['batch_size'])
        else:
            data_corpus_train = [' '.join(i) for i in dataset.get_corpus()]
            train_loader, self.id2word = self.preprocess(vocab, data_corpus_train, batch_size=self.hyperparameters['batch_size'])
            test_loader = None

        if self.model_name == 'DVAE_RSVITM':
            self.model = DVAE_RSVITM(vocab_size=len(vocab), topic_size=self.hyperparameters['num_topics'])
            
        elif self.model_name == 'DVAETM':
            self.model = DVAETM(vocab_size=len(vocab), topic_size=self.hyperparameters['num_topics'])
        
        elif self.model_name == 'DVAE_ProdLDA':
            self.model = DVAE_ProdLDA(vocab_size=len(vocab), topic_size=self.hyperparameters['num_topics'])
        
        
        trainer = Trainer(
            max_epochs=self.hyperparameters['num_epochs'],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=[0] if torch.cuda.is_available() else "cpu",
        )

        trainer.fit(model=self.model, train_dataloaders=train_loader)

        result = self.get_info()
        result['topic-document-matrix'] = self.get_doc_topic_distribution(train_loader)

        if test_loader is not None:
            result['test-topic-document-matrix'] = self.get_doc_topic_distribution(test_loader)

        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])

    def get_info(self):
        info = {}
        info['topic-word-matrix'], info['topics'] = self.get_beta_n_topics(self.id2word)
        return info
    
    def get_beta_n_topics(self, id2word):
        # load best model
        self.model.eval()
        self.model.freeze()

        # get beta
        beta = self.model.decoder.weight.detach().cpu().numpy().T
        topics = beta.argsort(axis=1)[:, ::-1]
        
        # top N words
        topics = topics[:, :self.top_word]
        topics = [[id2word[i] for i in topic] for topic in topics]
        return beta, topics
    
    def get_doc_topic_distribution(self, dataloader):
        self.model.eval()
        self.model.freeze()
        all_doc_topics = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch['bow'].float().to(self.model.device)
                if self.model_name == 'DVAE_RSVITM':
                    _, alpha = self.model(x)
                    doc_topic = alpha / alpha.sum(dim=1, keepdim=True)
                elif self.model_name == 'DVAETM':
                    _, dist = self.model(x)
                    doc_topic = dist.mean
                    
                all_doc_topics.append(doc_topic.cpu())

        return torch.cat(all_doc_topics).numpy().T  # shape: [num_topics, num_docs]


    @staticmethod
    def preprocess(vocab, train, test=None, validation=None, batch_size=128):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        id2word = {i: w for i, w in enumerate(vocab)}
        vec = CountVectorizer(vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)

        x_train = vec.transform(train)
        train_data = TorchDatasetBoW(x_train.toarray())
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            test_data = TorchDatasetBoW(x_test.toarray())
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

            x_valid = vec.transform(validation)
            valid_data = TorchDatasetBoW(x_valid.toarray())
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader, valid_loader, id2word
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            valid_data = TorchDatasetBoW(x_valid.toarray())
            valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
            
            return train_loader, valid_loader, id2word
        if test is not None and validation is None:
            x_test = vec.transform(test)
            test_data = TorchDatasetBoW(x_test.toarray())
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader, id2word
        if test is None and validation is None:
            return train_loader, id2word

class TorchDatasetBoW(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'bow': self.data[idx]}