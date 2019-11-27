from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, LSTM, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
import itertools
from tqdm import tqdm
import numpy as np
import unidecode
import string
import os
import re
MAXLEN = 30
BATCH_SIZE = 1024
NGRAM = 5


char_vietnamese = ['á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',]
char_vietnamese.extend([c.upper() for c in char_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(char_vietnamese)))
# print(len(alphabet))

HIDDEN_SIZE = 256

model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet)), return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

def encode(text, maxlen = MAXLEN):
    text = '\x00' + text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if(i < maxlen - 1):
        for j in range(i+1, maxlen):
            x[j, 0] = 1
    return x 

def decode(x, cal_argmax = True):
    if cal_argmax:
        x = x.argmax(axis=1)
    return ''.join(alphabet[i] for i in x)

def generate_data(data, batch_size=128):
    cur_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):  
            y.append(encode(data[cur_index]))
            x.append(encode(unidecode.unidecode(data[cur_index])))
            cur_index += 1
            
            if cur_index > len(data)-1:
                cur_index = 0
        
        yield np.array(x), np.array(y)

# list_ngrams = []
with open('/content/drive/My Drive/list_ngrams.txt', 'r') as f:
    list_ngrams = f.read().split('\n')

from sklearn.model_selection import train_test_split
train_data, valid_data = train_test_split(list_ngrams, test_size = 0.2, random_state = 2019)

train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
valid_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

checkpointer = ModelCheckpoint(filepath=os.path.join('./model_{val_loss:.4f}_{val_acc:.4f}.h5'), save_best_only=True, verbose=1)
early = EarlyStopping(patience=2, verbose=1)

model.fit_generator(train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=10,
                    validation_data=valid_generator, validation_steps=len(valid_data)//BATCH_SIZE,
                    callbacks=[checkpointer, early])


from keras.models import load_model
from format_data import gen_ngrams
from collections import collections

model = load_model("")

def extract_phrases(text):
    pattern = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(pattern, text)

def guess(ngram):
    text = ' '.join(ngram)
    preds = model.predict(np.array([encode(text)]), verbose=0)
    return decode(preds[0], calc_argmax=True).strip('\x00')

def add_accent(text):
    ngrams = list(gen_ngrams(text.lower(), n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [collections.Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates)
    return output

def accent_sentence(sentence):
  list_phrases = extract_phrases(sentence)
  output = ""
  for phrases in list_phrases:
    if len(phrases.split()) < 2 or not re.match("\w[\w ]+", phrases):
      output += phrases
    else:
      output += add_accent(phrases)
      if phrases[-1] == " ":
        output += " "
  return output