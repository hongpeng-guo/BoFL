##################################################################
# IMDB sentiment analysis with LSTM model
##################################################################
 
import torch
import torch.nn as nn
import torchtext.datasets as datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
 
DATASETS_ROOT = '/home/sl29/data/energyFLDatasets'

class RNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        embedded = self.dropout(self.embedding(text))  
        output, (hidden, cell) = self.rnn(embedded)   
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        return self.fc(hidden)


class IMDB_LSTM:
 
   ''' Apply LSTM on IMDB dataset for classification training'''
 
   def __init__(self, trainset_size, batch_size):
 
      self.trainlist = list(datasets.IMDB(root=DATASETS_ROOT, split='train'))[:trainset_size]
      self.max_len = 0
      self.get_vocab()
      self.get_dataloader(batch_size)

      self.model = RNN(vocab_size = self.vocab.__len__(), embedding_dim=100, 
                     hidden_dim=256, output_dim=1, n_layers=2, bidirectional=True, 
                     dropout=0.5, pad_idx=self.vocab["<pad>"])
      
      self.criterion = nn.BCEWithLogitsLoss()
      self.optimizer = torch.optim.Adam(self.model.parameters())

   def get_vocab(self):
      self.tokenizer = get_tokenizer('basic_english')

      def yield_tokens(data_iter):
         for _, text in data_iter:
            res = self.tokenizer(text)
            self.max_len = max(len(res), self.max_len)
            yield res

      self.vocab = build_vocab_from_iterator(yield_tokens(self.trainlist), specials=["<unk>", "<pad>"])
      self.vocab.set_default_index(self.vocab["<unk>"])

   def get_dataloader(self, batch_size):

      text_pipeline = lambda x: self.vocab(self.tokenizer(x))
      label_pipeline = lambda x: 1 if x == 'pos' else 0

      def collate_fn(batch):
         text_batch, label_batch = [], []
         for label_sample, text_sample in batch:
            label_batch.append(label_pipeline(label_sample))
            text_batch.append(torch.tensor(text_pipeline(text_sample)))

         pad_f = nn.ConstantPad1d((0, self.max_len - text_batch[0].shape[0]), self.vocab["<pad>"])
         text_batch[0] = pad_f(text_batch[0])
         text_batch = pad_sequence(text_batch, padding_value=self.vocab["<pad>"])

         return torch.tensor(label_batch, dtype=torch.float64).unsqueeze(1), text_batch
      
      self.trainloader = DataLoader(self.trainlist, batch_size=batch_size, collate_fn=collate_fn)


 
 
if __name__ == "__main__":
 
   task_model = IMDB_LSTM(320, 8)
   device = device = torch.device("cuda")
   model = task_model.model.to(device)

   # warmup
   print('start warmup')
   for idx, (label, text) in enumerate(task_model.trainloader):
      predicted_label = model(text.to(device))
      loss = task_model.criterion(predicted_label, label.to(device))
      loss.backward()
      task_model.optimizer.step()

   import time, os, sys, inspect

   currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
   parentdir = os.path.dirname(currentdir)
   sys.path.insert(0, parentdir)
   import powerLog as pl
   from powerLog import PowerLogger

   logger = PowerLogger(interval=0.1, nodes=list(filter(lambda n: n[0].startswith('module/'), pl.getNodes())))

   print('start train')
   logger.start()
   for _ in range(3):
      for idx, (label, text) in enumerate(task_model.trainloader):
         t0 = time.time()
         label, text = label.to(device), text.to(device)
         predicted_label = model(text)
         loss = task_model.criterion(predicted_label, label)
         loss.backward()
         task_model.optimizer.step()
         torch.cuda.synchronize()
         t1 = time.time()
         print(predicted_label.size(), t1 - t0)
      time.sleep(10)
   logger.stop()

   logger.showDataTraces()

      
      
 
 

