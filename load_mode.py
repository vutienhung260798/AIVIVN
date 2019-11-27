from keras.models import load_model
from format_data import gen_ngrams
from collections import Counter
import re
import numpy as np
import unidecode
import string
# import numpy as np
import json
NGRAM = 5
MAXLEN = 30

char_vietnamese = ['á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',]
char_vietnamese.extend([c.upper() for c in char_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits + ''.join(char_vietnamese)))

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

model = load_model("/home/hung-vt/AIVIVN/model_0.0490_0.9841.h5")

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
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates if candidates != '')
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

print("san pham nay khong tot")

def add_accent_sentence(input_file_json = '', output_file_json = 'add_accent.json'):
    try:
        with open(input_file_json, 'r') as f_r:
            data = f_r.read()
        obj = json.loads(data)
        sentences = obj['content']

        change_sentences = []

        for sentence in sentences:
            content = sentence['content']
            content = accent_sentence(unidecode.unidecode(content))
            change_sentences.append({'id': sentence['id'], 'content': content, 'label': sentence['label']})
        
        add_accent = {'sentence': change_sentences}

        with open(output_file_json, 'w', encoding='utf-8') as f:
            json.dump(add_accent, f, indent =4, ensure_ascii=False)
    
    except FileNotFoundError:
        add_accent.close()
        print('Path File Sentences is NOT exist')

