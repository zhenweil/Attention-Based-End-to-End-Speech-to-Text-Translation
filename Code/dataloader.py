import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils import rnn
is_mini = False
def load_data(PATH):
    speech_train = np.load(PATH + 'train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load(PATH + 'dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load(PATH + 'test.npy', allow_pickle=True, encoding='bytes')
    transcript_train = np.load(PATH + 'train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load(PATH + 'dev_transcripts.npy', allow_pickle=True,encoding='bytes')
    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid

'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter2index, index2letter = create_dictionaries(letter_list)
    transcript_list = []
    for sentense in transcript:
        sentense_list = []
        for word in sentense:
            for char_num in word:
                if(is_mini == False):
                    sentense_list.append(letter2index[chr(char_num)])
                else:
                    sentense_list.append(letter2index[char_num])
            sentense_list.append(letter2index[' '])
        sentense_list.pop()
        sentense_list.append(letter2index['<eos>'])
        transcript_list.append(sentense_list)
    return transcript_list


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    length = len(letter_list)
    for i in range(length):
        letter2index[letter_list[i]] = i
        index2letter[i] = letter_list[i]
    return letter2index, index2letter

class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        self.text = text

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, index):
        if (self.isTrain == True and self.text is not None):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    x = [i[0] for i in batch_data]
    y = [i[1] for i in batch_data]
    x_len = [len(i) for i in x]
    y_len = [len(i) for i in y]
    x_len = torch.LongTensor(x_len)
    y_len = torch.LongTensor(y_len)
    x = rnn.pad_sequence(x)
    y = rnn.pad_sequence(y, batch_first = True)
    return x, x_len, y, y_len

def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    x = [i for i in batch_data]
    x_len = [len(i) for i in x]
    x_len = torch.LongTensor(x_len)
    x = rnn.pad_sequence(x)
    return x, x_len