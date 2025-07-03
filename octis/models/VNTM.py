from octis.models.model import AbstractModel
from octis.models.vontss.model import TopicModel

class VNTM(AbstractModel):

    def __init__(self, num_topics=10, num_epochs=100,
                 use_partitions=False, use_validation=False, 
                 ):

        assert not(use_validation or use_partitions), "VNTM does not consider splitting the dataset'."
        
        super(VNTM, self).__init__()
        self.hyperparameters = dict()
        self.hyperparameters['num_topics'] = int(num_topics)
        self.hyperparameters['num_epochs'] = int(num_epochs)
        
        self.use_partitions = use_partitions
        self.use_validation = use_validation
        self.model = None

    def train_model(self, dataset, hyperparameters=None, top_words=10):

        if hyperparameters is None:
            hyperparameters = {}

        self.set_params(hyperparameters)
        self.top_word = top_words

        if self.use_partitions and not self.use_validation:
            train, test = dataset.get_partitioned_corpus(use_validation=False)
            data_corpus_train = [' '.join(i) for i in train]
            data_corpus_test = [' '.join(i) for i in test]

            self.model = TopicModel(
                epochs=self.hyperparameters['num_epochs'],
                embedding_dim=self.hyperparameters['num_topics'],
                top_n_words=self.top_word
            )
            result = self.model.train_model(dataset=data_corpus_train)
            result['test-topic-document-matrix'] = self.model.test_model(data_corpus_test)
        
        else:
            texts = [' '.join(i) for i in dataset.get_corpus()]

            self.model = TopicModel(
                epochs=self.hyperparameters['num_epochs'],
                embedding_dim=self.hyperparameters['num_topics'],
                top_n_words=self.top_word
            )
            result = self.model.train_model(dataset=texts)
       
        return result

    def set_params(self, hyperparameters):
        for k in hyperparameters.keys():
            if k in self.hyperparameters.keys():
                self.hyperparameters[k] = hyperparameters.get(k, self.hyperparameters[k])
