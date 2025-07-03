
import torch
from torch.utils.data import Dataset

class DocDataset(Dataset):
    def __init__(self, bows, docs, dictionary):
        self.bows = bows
        self.docs = docs
        self.dictionary = dictionary
        self.vocabsize = len(dictionary)
        
    def __getitem__(self,idx):
        bow = torch.zeros(self.vocabsize)
        item = list(zip(*self.bows[idx])) # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt,bow
    
    def __len__(self):
        return len(self.bows)
    
    def collate_fn(self,batch_data):
        texts,bows = list(zip(*batch_data))
        return texts,torch.stack(bows,dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc