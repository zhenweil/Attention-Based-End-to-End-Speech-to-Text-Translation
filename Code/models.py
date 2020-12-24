import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.utils import rnn
from dataloader import create_dictionaries
from util import plot_attn_flow
import random
from regularization import LockedDropout
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, query, key, value, lens):
        '''
        :param query :(batch_size, hidden_size) Query is the output of LSTMCell from Decoder
        :param keys: (batch_size, max_len, encoder_size) Key Projection from Encoder
        :param values: (batch_size, max_len, encoder_size) Value Projection from Encoder
        :return context: (batch_size, encoder_size) Attended Context
        :return attention_mask: (batch_size, max_len) Attention mask that can be plotted 
        '''
        query = torch.unsqueeze(query, dim = 2) # (batch_size, encoder_size, 1)
        key = key.permute(1, 0, 2) # (batch_size, max_len, encoder_size)
        energy = torch.bmm(key, query) # (batch_size, max_len, 1)
        energy = torch.squeeze(energy, dim = 2) # (batch_size, max_len)
        # apply mask
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1)
        mask = mask.to(DEVICE)
        energy.masked_fill_(mask, float('-inf'))
        attention = self.softmax(energy) # (batch_size, max_len)
        attention_expanded = torch.unsqueeze(attention, dim = 2) # (batch_size, max_len, 1)
        value = value.permute(1, 2, 0) # (batch_size, encoder_size, max_len)
        context = torch.bmm(value, attention_expanded).squeeze(2) # shape: (batch_size, encoder_size, 1)
        return context, attention

class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.ld = LockedDropout()
        #print(self.blstm._all_weights)
        #self.blstm = WeightDrop(self.blstm, ['weight_hh_l0'], dropout = 0.2)

    def forward(self, x):
        x_unpacked, x_lens = rnn.pad_packed_sequence(x) # convert it to unpacked sequence
        x_unpacked = self.ld(x_unpacked, dropout = 0.2)
        x_even = x_unpacked[:(x_unpacked.shape[0]//2)*2,:,:]
        x_reshaped = x_even.permute(1, 0, 2)
        x_reshaped = x_reshaped.reshape(x_unpacked.shape[1], x_unpacked.shape[0]//2, x_unpacked.shape[2]*2)
        x_reshaped = x_reshaped.permute(1, 0, 2)
        x_lens = x_lens//2
        x_reshaped_packed = rnn.pack_padded_sequence(x_reshaped, x_lens, enforce_sorted = False)
        embedding, _ = self.blstm(x_reshaped_packed)
        return embedding
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size,key_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pBLSTM = nn.Sequential(pBLSTM(hidden_dim*4, hidden_dim),
                                    pBLSTM(hidden_dim*4, hidden_dim),
                                    pBLSTM(hidden_dim*4, hidden_dim))
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)
        self.ld = LockedDropout()
    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        outputs = self.pBLSTM(outputs)
        linear_input, lens = utils.rnn.pad_packed_sequence(outputs)
        linear_input = self.ld(linear_input, dropout = 0.2)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)
        return keys, value, lens


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size, key_size, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=512)
        self.lstm2 = nn.LSTMCell(input_size=512, hidden_size=key_size)
        self.context_linear = nn.Linear(value_size, value_size)
        self.key_linear = nn.Linear(key_size, key_size)
        self.isAttended = True
        self.value_size = value_size
        self.attention = Attention()
        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self, epoch, batch_idx, key, values, lens, text=None, isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        # input shape is: (max_len, batch_size, feature_size)
        plot = []
        batch_size = key.shape[1]
        value_size = key.shape[2]
        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text) 
        else:
            max_len = 600
        predictions = []
        prediction = None
        hidden_states = [None, None]
        context = values[0,:,:]
        if(epoch < 10):
            tf_rate = 0.1 
        elif(epoch >= 10 and epoch < 20):
            tf_rate = 0.15
        elif(epoch >= 20 and epoch < 25):
            tf_rate = 0.2
        elif(epoch >= 25 and epoch < 30):
            tf_rate = 0.25
        elif(epoch >= 30 and epoch < 35):
            tf_rate = 0.3
        elif(epoch >= 35 and epoch < 40):
            tf_rate = 0.35
        else:
            tf_rate = 0.4
        for i in range(max_len):
            if(random.random() > tf_rate):
                tf = True
            else:
                tf = False

            if(i == 0):
                start_of_sentence = torch.full((batch_size, 1), 33, dtype = torch.long).to(DEVICE)
                char_embed = self.embedding(start_of_sentence).squeeze(dim = 1)
            else:
                if(isTrain == True and tf == True):
                    char_embed = embeddings[:, i-1, :].to(DEVICE)
                else:
                    prediction_prob = self.softmax(prediction)
                    char_embed = self.embedding(prediction_prob.argmax(dim = -1)).squeeze(dim = 1)
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])
            output = hidden_states[1][0]

            ### Compute attention from the output of the second LSTM Cell ###
            if(self.isAttended == True):
                context, attention = self.attention(output, key, values, lens)
                plot.append(attention[0,:])
            else:
                context = torch.zeros(batch_size, self.value_size).to(DEVICE)
            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))

        if(batch_idx % 50 == 0 and isTrain == True):
            attention_plot = torch.stack(plot, dim = 0).detach().to("cpu")
            path = "../Attention/attention.png"
            plot_attn_flow(attention_plot, path)
        return torch.cat(predictions, dim = 1)
            
class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size, key_size, isAttended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, value_size, key_size)
        self.decoder = Decoder(vocab_size, hidden_dim, value_size, key_size)

    def forward(self, epoch, batch_idx, speech_input, speech_len, text_input=None, isTrain=True):
        key, value, lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(epoch, batch_idx, key, value, lens, text_input)
        else:
            predictions = self.decoder(epoch, batch_idx, key, value, lens, text=None, isTrain=False)
        return predictions