import string
import re
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

def preprocess(sent, exclude, sp_tokens=False):
    '''
    PARAMETERS
    ----------
    sent (str): sentence to preprocess
    exclude (str): characters to exclude (like punctuations)
    sp_tokens (bool): If True, special tokens '<start>' and '<end>'
                      will be added. Default False.
    '''
    
    sent = sent.lower()
    sent = re.sub("'", '', sent)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    sent = sent.strip()
    sent = re.sub(" +", " ", sent)
    if sp_tokens:
        sent = '<start> ' + sent + ' <end>'
    
    return sent


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

            
def max_length(tensor):
    return max(len(t) for t in tensor)


def data_loader(path, batch_size, samples=None, max_len=None, reverse=False):
    '''
    PARAMTERS
    ---------
    path: path to the translation file
    
    '''
    
    # loading the dataset
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    lines = [l.split('\t')[:-1] for l in lines]
    
    if samples:
        random.shuffle(lines)
        lines = lines[:samples]
    
    if reverse:
        lines = [l[::-1] for l in lines]
        
    exclude = set(string.punctuation)
    
    # preprocessing
    prep_lines = [
        [preprocess(i, exclude, sp_tokens=False), 
         preprocess(j, exclude, sp_tokens=True)]
        for i, j in lines
    ]
    
    # Language index (vocab, idx2word, word2idx)
    inp_lang = LanguageIndex(en for en, ma in prep_lines)
    tgt_lang = LanguageIndex(ma for en, ma in prep_lines)
    
    # creating input and target tensor
    input_tensor = [[inp_lang.word2idx[w] for w in inp.split(' ')] 
                    for inp, tgt in prep_lines]
    
    target_tensor = [[tgt_lang.word2idx[w] for w in tgt.split(' ')] 
                     for inp, tgt in prep_lines]
    
    
    # calculating max length for padding
    if max_len:
        new_input_tensor = []
        new_target_tensor = []
        for i, j in zip(input_tensor, target_tensor):
            if (len(i) <= max_len) and (len(j) <= max_len):
                new_input_tensor.append(i)
                new_target_tensor.append(j)

        input_tensor = new_input_tensor
        target_tensor = new_target_tensor
        max_length_inp = max_length_tgt = max_len

    else:
        max_length_inp, max_length_tgt = max_length(input_tensor), max_length(target_tensor)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor, 
        maxlen=max_length_inp,
        padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor, 
        maxlen=max_length_tgt, 
        padding='post')

    (input_tensor_train, 
     input_tensor_val, 
     target_tensor_train, 
     target_tensor_val) = train_test_split(
         input_tensor, 
         target_tensor, 
         test_size=0.1, 
         random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)
        ).shuffle(len(input_tensor_train)).batch(batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)
        ).shuffle(len(input_tensor_val)).batch(batch_size, drop_remainder=True)
    
    return train_dataset, test_dataset, inp_lang, tgt_lang, max_length_inp, max_length_tgt
