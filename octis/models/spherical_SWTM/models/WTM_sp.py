import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from octis.models.spherical_SWTM.models.wae_sp import WAE
from octis.models.spherical_SWTM.utils import evaluate_topic_quality, smooth_curve
import random


class S2WTM:
    def __init__(self, bow_dim=10000, n_topic=20, device=None,
                 taskname=None, dropout=0.0, batch_size=256, learning_rate=1e-3,
                 num_epochs=100, log_every=5, beta=1.0, dist='gmm_std', loss_type='sph_sw',
                 num_projections=100, ftype='linear', degree=3, p=2
                 ):
        self.bow_dim = bow_dim
        self.n_topic = n_topic
        
        self.device = device
        self.id2token = None
        self.dist = dist
        self.dropout = dropout
        self.taskname = taskname

        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.log_every = log_every
        self.beta = beta

        self.loss_type = loss_type
        self.num_projections = num_projections
        self.ftype = ftype
        self.degree = degree
        self.p = p

        self.wae = WAE(encode_dims=[bow_dim, 1024, 512, n_topic], decode_dims=[n_topic, 512, bow_dim],
                       dropout=dropout, nonlin='relu', dist=self.dist, batch_size=self.batch_size)
        
        if device != None:
            self.wae = self.wae.to(device)

    def train(self, train_data, test_data=None, 
              ckpt=None, verbose=False, topK=10):
        if verbose:
            print("Settings: \n\
                   bow_dim: {}\n\
                   n_topic: {}\n\
                   dist: {}\n\
                   dropout: {}\n\
                   batch_size: {}\n\
                   learning_rate: {}\n\
                   num_epochs: {}\n\
                   log_every: {}\n\
                   beta: {}\n\
                   loss_type: {}\n\
                   num_projections: {}\n\
                   ftype: {}\n\
                   degree: {}\n\
                   p: {}".format(
                self.bow_dim, self.n_topic, self.dist, self.dropout,
                self.batch_size, self.learning_rate, self.num_epochs,
                self.log_every, self.beta, self.loss_type,
                self.num_projections, self.ftype, self.degree, self.p))
        self.wae.train()
        self.id2token = {v: k for k,v in train_data.dictionary.token2id.items()}
        data_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=train_data.collate_fn
            )

        optimizer = torch.optim.Adam(self.wae.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        if ckpt:
            self.load_model(ckpt["net"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
        else:
            start_epoch = 0

        trainloss_lst, valloss_lst = [], []
        c_v_lst, c_w2v_lst, c_uci_lst, c_npmi_lst, mimno_tc_lst, td_lst = [], [], [], [], [], []
        for epoch in range(start_epoch, self.num_epochs):
            epochloss_lst = []
            for iter, data in enumerate(data_loader):
                optimizer.zero_grad()

                txts, bows = data
                bows = bows.to(self.device)

                bows_recon, theta_q = self.wae(bows)
                theta_prior = self.wae.sample(ori_data=bows).to(self.device)

                logsoftmax = torch.log_softmax(bows_recon, dim=1)
                rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                
                if torch.isnan(theta_q).any() or torch.isnan(bows).any():
                    print("NaN detected in theta_q or bows")
                
                if self.loss_type=='sw':
                    ot_loss = self.wae.swd_loss(theta_q,
                                                theta_prior,
                                                num_projections=self.num_projections,
                                                device=self.device,
                                                p=self.p,
                                                ftype=self.ftype,
                                                degree=self.degree
                                                )
                elif self.loss_type=='sph_sw':
                    ot_loss = self.wae.sp_swd_loss(theta_q,
                                                   theta_prior,
                                                   num_projections=self.num_projections,
                                                   device=self.device,
                                                   p=self.p)
                elif self.loss_type=='sph_sw_uni':
                    ot_loss = self.wae.sp_swd_unif_loss(theta_q,
                                                        num_projections=self.num_projections,
                                                        device=self.device
                                                        )
                
                else:
                    raise Exception('The following OT-based loss: {} not implemented'.format(self.loss_type))

                s = torch.sum(bows)/len(bows)
                lamb = (5.0*s*torch.log(torch.tensor(1.0 *bows.shape[-1]))/torch.log(torch.tensor(2.0)))
                ot_loss = ot_loss * lamb

                loss = rec_loss + ot_loss * self.beta

                loss.backward()
                optimizer.step()

                trainloss_lst.append(loss.item()/len(bows))
                epochloss_lst.append(loss.item()/len(bows))
                if verbose and ((iter+1) % 10 == 0):
                    print('Epoch {:>3d}\tIter {:>4d}\tLoss:{:.7f}\tRec Loss:{:.7f}\tOT-Loss:{:.7f}'.format(
                        epoch+1,
                        iter+1,
                        loss.item()/len(bows),
                        rec_loss.item()/len(bows),
                        ot_loss.item()/len(bows)))
            #scheduler.step()
            if (epoch+1) % self.log_every == 0:
                save_name = f'./ckpt/S2WTM_{self.taskname}_tp{self.n_topic}_{self.dist}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}_{epoch+1}.ckpt'
                checkpoint = {
                    "net": self.wae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "param": {
                        "bow_dim": self.bow_dim,
                        "n_topic": self.n_topic,
                        "taskname": self.taskname,
                        "dist": self.dist,
                        "dropout": self.dropout
                    }
                }
                torch.save(checkpoint,save_name)
                print(f'Epoch {(epoch+1):>3d}\tLoss:{sum(epochloss_lst)/len(epochloss_lst):<.7f}')
                res = {}
                res['topics'] = self.show_topic_words(topK=topK)
                print('='*30)
                if test_data!=None:
                    c_v,c_w2v,c_uci,c_npmi,mimno_tc, td = self.evaluate(test_data,calc4each=False)
                    c_v_lst.append(c_v), c_w2v_lst.append(c_w2v), c_uci_lst.append(c_uci),c_npmi_lst.append(c_npmi), mimno_tc_lst.append(mimno_tc), td_lst.append(td)

    def evaluate(self, test_data, calc4each=False):
        topic_words = self.show_topic_words()
        return evaluate_topic_quality(topic_words, test_data, taskname=self.taskname, calc4each=calc4each)


    def inference_by_bow(self, doc_bow):
        # doc_bow: torch.tensor [vocab_size]; optional: np.array [vocab_size]
        if isinstance(doc_bow,np.ndarray):
            doc_bow = torch.from_numpy(doc_bow)
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            self.wae.eval()
            theta = F.softmax(self.wae.encode(doc_bow),dim=1)
            return theta.detach().cpu().numpy()


    def inference(self, doc_tokenized, dictionary,normalize=True):
        doc_bow = torch.zeros(1,self.bow_dim)
        for token in doc_tokenized:
            try:
                idx = dictionary.token2id[token]
                doc_bow[0][idx] += 1.0
            except:
                print(f'{token} not in the vocabulary.')
        doc_bow = doc_bow.to(self.device)
        with torch.no_grad():
            self.wae.eval()
            theta = self.wae.encode(doc_bow)
            if normalize:
                theta = F.softmax(theta,dim=1)
            return theta.detach().cpu().squeeze(0).numpy()
    

    def get_doc_topic_distribution(self, dataset, n_samples=20):
        self.wae.eval()
        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=dataset.collate_fn,
                                 worker_init_fn=S2WTM.seed_worker
                                 )
        
        final_thetas = []
        for _ in range(n_samples):
            with torch.no_grad():
                collect_theta = []
                for iter, data in enumerate(data_loader):
                    txts, bows = data
                    bows = bows.to(self.device)
                    theta = self.wae.encode(bows)
                    collect_theta.extend(theta.cpu().numpy().tolist())
                
                final_thetas.append(np.array(collect_theta))     
        return np.sum(final_thetas, axis=0) / n_samples

    def get_embed(self,train_data, num=1000):
        self.wae.eval()
        data_loader = DataLoader(train_data, batch_size=512,shuffle=False, num_workers=4, collate_fn=train_data.collate_fn)
        embed_lst = []
        txt_lst = []
        cnt = 0
        for data_batch in data_loader:
            txts, bows = data_batch
            embed = self.inference_by_bow(bows)
            embed_lst.append(embed)
            txt_lst.append(txts)
            cnt += embed.shape[0]
            if cnt>=num:
                break
        embed_lst = np.concatenate(embed_lst,axis=0)[:num]
        txt_lst = np.concatenate(txt_lst,axis=0)[:num]
        return txt_lst, embed_lst

    def get_topic_word_dist(self,normalize=True):
        self.wae.eval()
        with torch.no_grad():
            idxes = torch.eye(self.n_topic).to(self.device)
            word_dist = self.wae.decode(idxes)  # word_dist: [n_topic, vocab.size]
            if normalize:
                word_dist = F.softmax(word_dist,dim=1)
            return word_dist.detach().cpu().numpy()

    def show_topic_words(self, topic_id=None, topK=15, dictionary=None):
        self.wae.eval()
        topic_words = []
        idxes = torch.eye(self.n_topic).to(self.device)
        word_dist = self.wae.decode(idxes)
        word_dist = F.softmax(word_dist, dim=1)
        vals, indices = torch.topk(word_dist, topK, dim=1)
        vals = vals.cpu().tolist()
        indices = indices.cpu().tolist()
        if self.id2token==None and dictionary!=None:
            self.id2token = {v:k for k,v in dictionary.token2id.items()}
        if topic_id == None:
            for i in range(self.n_topic):
                topic_words.append([self.id2token[idx] for idx in indices[i]])
        else:
            topic_words.append([self.id2token[idx] for idx in indices[topic_id]])
        return topic_words
    
    def load_model(self, model):
        self.wae.load_state_dict(model)
    
    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

