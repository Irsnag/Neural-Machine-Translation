import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from nltk import word_tokenize



class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)

    def forward(self, input):
        embed = self.embedding(input)
        hs, _ = self.rnn(embed)
        return hs


class seq2seqAtt(nn.Module):
  
  def __init__(self, hidden_dim, hidden_dim_s, hidden_dim_t):
      super(seq2seqAtt, self).__init__()
      self.ff_concat = nn.Linear(hidden_dim_s+hidden_dim_t, hidden_dim)
      self.ff_score = nn.Linear(hidden_dim, 1, bias=False) 

  def forward(self, target_h, source_hs):
      target_h_rep = target_h.repeat(source_hs.size(0), 1, 1) 
      concat_output = torch.tanh((self.ff_concat(torch.cat((target_h_rep, source_hs), dim=2))))
      scores = self.ff_score(concat_output) 
      scores = scores.squeeze(dim=2) 
      norm_scores = torch.softmax(scores, 0)
      source_hs_p = source_hs.permute((2, 0, 1)) 
      weighted_source_hs = (norm_scores * source_hs_p) 
      ct = torch.sum(weighted_source_hs.permute((1, 2, 0)), 0, keepdim=True)
      return ct, norm_scores


class Decoder(nn.Module):
  
    def __init__(self, vocab_size, embedding_dim, hidden_dim, padding_idx):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.ff_concat = nn.Linear(2*hidden_dim, hidden_dim)
        self.predict = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, source_context, h):
        embedded_input = self.embedding(input)
        hs, h = self.rnn(embedded_input, h)
        tilde_h = torch.tanh(self.ff_concat(torch.cat((source_context, h), dim=2)))
        prediction = self.predict(tilde_h)
        return prediction, h  


class seq2seqModel(nn.Module):
    '''the full seq2seq model'''
    ARGS = ['vocab_s','source_language','vocab_t_inv','embedding_dim_s','embedding_dim_t',
            'hidden_dim_s','hidden_dim_t','hidden_dim_att','do_att','padding_token',
            'oov_token','sos_token','eos_token','max_size']
    def __init__(self, vocab_s, source_language, vocab_t_inv, embedding_dim_s, embedding_dim_t,
                 hidden_dim_s, hidden_dim_t, hidden_dim_att, do_att, padding_token,
                 oov_token, sos_token, eos_token, max_size):
        super(seq2seqModel, self).__init__()
        self.vocab_s = vocab_s
        self.source_language = source_language
        self.vocab_t_inv = vocab_t_inv
        self.embedding_dim_s = embedding_dim_s
        self.embedding_dim_t = embedding_dim_t
        self.hidden_dim_s = hidden_dim_s
        self.hidden_dim_t = hidden_dim_t
        self.hidden_dim_att = hidden_dim_att
        self.do_att = do_att 
        self.padding_token = padding_token
        self.oov_token = oov_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_size = max_size

        self.max_source_idx = max(list(vocab_s.values()))
        print('max source index', self.max_source_idx)
        print('source vocab size', len(vocab_s))

        self.max_target_idx = max([int(elt) for elt in list(vocab_t_inv.keys())])
        print('max target index', self.max_target_idx)
        print('target vocab size', len(vocab_t_inv))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(self.max_source_idx + 1, self.embedding_dim_s, self.hidden_dim_s, self.padding_token).to(self.device)
        self.decoder = Decoder(self.max_target_idx + 1, self.embedding_dim_t, self.hidden_dim_t, self.padding_token).to(self.device)

        if self.do_att:
            self.att_mech = seq2seqAtt(self.hidden_dim_att, self.hidden_dim_s, self.hidden_dim_t).to(self.device)

    def my_pad(self, my_list):
        '''my_list is a list of tuples of the form [(tensor_s_1,tensor_t_1),...,(tensor_s_batch,tensor_t_batch)]
        the <eos> token is appended to each sequence before padding
        https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_sequence'''
        batch_source = pad_sequence([torch.cat((elt[0], torch.LongTensor([self.eos_token]))) for elt in my_list],
                                    batch_first=True, padding_value=self.padding_token)
        batch_target = pad_sequence([torch.cat((elt[1], torch.LongTensor([self.eos_token]))) for elt in my_list],
                                    batch_first=True, padding_value=self.padding_token)
        return batch_source, batch_target

    def forward(self, input, max_size, is_prod):

        if is_prod:
            input = input.unsqueeze(1)  # (seq) -> (seq,1) 1D input <=> we receive just one sentence as input (predict/production mode)
        current_batch_size = input.size(1)

        source_hs = self.encoder(input)
        target_h = torch.zeros(size=(1, current_batch_size, self.hidden_dim_t)).to(self.device)  # init (1,batch,feat)

 
        target_input = torch.LongTensor([2]).repeat(current_batch_size).unsqueeze(0).to(self.device)  # init (1,batch)
        pos = 0
        eos_counter = 0
        logits = []
        scores = []

        while True:
            if self.do_att:
                source_context, score = self.att_mech(target_h, source_hs)  # (1,batch,feat)
                scores.append(score)
            else:
                source_context = source_hs[-1, :, :].unsqueeze(0)  # (1,batch,feat) last hidden state of encoder
       
            prediction, target_h = self.decoder(target_input, source_context, target_h)
            logits.append(prediction)  
            _, target_input = torch.max(prediction, 2)
            eos_counter += torch.sum(target_input == self.eos_token).item()
            pos += 1
            if pos >= max_size or (eos_counter == current_batch_size and is_prod):
                break
        to_return = torch.cat(logits, 0)  # logits is a list of tensors -> (seq,batch,vocab)
        if is_prod:
            to_return = to_return.squeeze(dim=1)  # (seq,vocab)
        return to_return, scores

    def fit(self, trainingDataset, testDataset, lr, batch_size, n_epochs, patience):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(parameters, lr=lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=self.padding_token) 

        train_loader = data.DataLoader(trainingDataset, batch_size=batch_size,
                                       shuffle=True, collate_fn=self.my_pad)  # returns (batch,seq)
        test_loader = data.DataLoader(testDataset, batch_size=512,
                                      collate_fn=self.my_pad)
        tdqm_dict_keys = ['loss', 'test loss']
        tdqm_dict = dict(zip(tdqm_dict_keys, [0.0, 0.0]))
        patience_counter = 1
        patience_loss = 99999
        for epoch in range(n_epochs):
            with tqdm(total=len(train_loader), unit_scale=True, postfix={'loss': 0.0, 'test loss': 0.0},
                      desc="Epoch : %i/%i" % (epoch, n_epochs - 1), ncols=100) as pbar:
                for loader_idx, loader in enumerate([train_loader, test_loader]):
                    total_loss = 0
                    if loader_idx == 0:
                        self.train()
                    else:
                        self.eval()
                    for i, (batch_source, batch_target) in enumerate(loader):
                        batch_source = batch_source.transpose(1, 0).to(
                            self.device)  # RNN needs (seq,batch,feat) but loader returns (batch,seq)
                        batch_target = batch_target.transpose(1, 0).to(self.device)  # (seq,batch)

                        # are we using the model in production / as an API?
                        is_prod = len(batch_source.shape) == 1  # if False, 2D input (seq,batch), i.e., train or test

                        if is_prod:
                            max_size = self.max_size
                            self.eval()
                        else:
                            max_size = batch_target.size(
                                0) 

                        unnormalized_logits, _ = self.forward(batch_source, max_size, is_prod)

                        sentence_loss = criterion(unnormalized_logits.flatten(end_dim=1), batch_target.flatten())

                        total_loss += sentence_loss.item()

                        tdqm_dict[tdqm_dict_keys[loader_idx]] = total_loss / (i + 1)

                        pbar.set_postfix(tdqm_dict)

                        if loader_idx == 0:
                            optimizer.zero_grad()  # flush gradient attributes
                            sentence_loss.backward()  # compute gradients
                            optimizer.step()  # update
                            pbar.update(1)

            if total_loss > patience_loss:
                patience_counter += 1
            else:
                patience_loss = total_loss
                patience_counter = 1  # reset

            if patience_counter > patience:
                break

    def sourceNl_to_ints(self, source_nl):
        '''converts natural language source sentence into source integers'''
        source_nl_clean = source_nl.lower().replace("'", ' ').replace('-', ' ')
        source_nl_clean_tok = word_tokenize(source_nl_clean, self.source_language)
        source_ints = [int(self.vocab_s[elt]) if elt in self.vocab_s else \
                           self.oov_token for elt in source_nl_clean_tok]

        source_ints = torch.LongTensor(source_ints).to(self.device)
        return source_ints

    def targetInts_to_nl(self, target_ints):
        '''converts integer target sentence into target natural language'''
        return ['<PAD>' if elt == self.padding_token else '<OOV>' if elt == self.oov_token \
            else '<EOS>' if elt == self.eos_token else '<SOS>' if elt == self.sos_token \
            else self.vocab_t_inv[elt] for elt in target_ints]

    def predict(self, source_nl):
        source_ints = self.sourceNl_to_ints(source_nl)
        logits, _ = self.forward(source_ints, self.max_size, True)  # (seq) -> (<=max_size,vocab)
        target_ints = logits.argmax(-1).squeeze()  # (<=max_size,1) -> (<=max_size)
        target_nl = self.targetInts_to_nl(target_ints.tolist())
        return ' '.join(target_nl)

    def save(self, path_to_file):
        attrs = {attr: getattr(self, attr) for attr in self.ARGS}
        attrs['state_dict'] = self.state_dict()
        torch.save(attrs, path_to_file)

    @classmethod 
    def load(cls, path_to_file):
        attrs = torch.load(path_to_file, map_location=lambda storage, loc: storage)  # allows loading on CPU a model trained on GPU, see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/6
        state_dict = attrs.pop('state_dict')
        new = cls(**attrs)  
        new.load_state_dict(state_dict)
        return new

    def attention_matrix(self, source):
        source_ints = self.sourceNl_to_ints(source)
        logits, scores = self.forward(source_ints, self.max_size, True)
        target_ints = logits.argmax(-1).squeeze()
        target_nl = self.targetInts_to_nl(target_ints.tolist())
        fig = plt.figure(figsize=(6, 7))
        ax = fig.add_subplot(111)
        scores[:6] = [tensor.detach().cpu().numpy() for tensor in scores[:6]]
        cax = ax.matshow(np.array(scores[:6]))
        fig.colorbar(cax)
        source_arr = source.split()
        ax.set_xticklabels([''] + [x for x in source_arr])
        ax.set_yticklabels([''] + [x for x in target_nl])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
