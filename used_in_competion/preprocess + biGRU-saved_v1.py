
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
embeding_file_path = "glove.840B.300d.txt"


# In[3]:


train['comment_text'][3]


# preprocessing :
# 1. lowercase
# 2. stopwords
# 3. low frequency words drop out
# 4. lemmatizer

# In[4]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
wl = WordNetLemmatizer()

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

def normalize(text):
    text = text.lower()
    translate_map = str.maketrans(filters, " " * len(filters))
    text = text.translate(translate_map)
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    stop_words = set(stopwords.words('english'))
    seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags if t[0] not in stop_words]
    #seq= [wl.lemmatize(t[0], pos=get_wordnet_pos(t[1])) for t in tags]

    return seq
#s = normalize(train["comment_text"][3/


# In[5]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 100000
embed_size = 300
maxlen = 150

data = pd.concat((train['comment_text'], test['comment_text']))
seqs = [normalize(text) for text in data]

def seq_to_sequence(seq, word_index):
    sequence = []
    for word in seq:
        if not word_index.get(word): continue 
        sequence.append(word_index[word])
    return sequence

def fit_on_sequence(seqs):
    word_counts = dict()
    for seq in seqs:
        for w in seq:
            if w not in word_counts:
                word_counts[w] = 0
            word_counts[w] += 1
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts if wc[1]>=3]
    sorted_voc = sorted_voc[:max_features]
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
    return word_index
    
word_index = fit_on_sequence(seqs)
train_words = [seq_to_sequence(seq, word_index) for seq in seqs[:train.shape[0]]]
test_words = [seq_to_sequence(seq, word_index) for seq in seqs[train.shape[0]:]]
train_words = pad_sequences(train_words, maxlen=maxlen )
test_words = pad_sequences(test_words, maxlen=maxlen)


# In[6]:


def get_coef(word, *coefs):
    return word, np.asarray(coefs, dtype=np.float32)
embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path))


# In[7]:


max_words = min(max_features, len(word_index))
embeding_matrix = np.zeros((max_words+1, embed_size))
lose = 0
for word,i in word_index.items():
    if word not in embeding_dict: 
        continue
    if i>max_words: 
        lose += 1
        continue 
    embeding_matrix[i] = embeding_dict[word]
print(lose)


# In[8]:


len(word_index), max_features


# In[10]:


from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D,Dropout
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    x = Embedding(max_words+1, embed_size, weights=[embeding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    x,s_h = Bidirectional(GRU(80, return_sequences=True, return_state=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool,max_pool, s_h])
    hidden = Dense(50, activation='tanh')(conc)
    h = Dropout(0.3)(hidden)
    oup = Dense(6, activation='sigmoid')(h)
    
    model = Model(input=inp, output=oup)
    model.compile(loss='binary_crossentropy',optimizer = Adam(lr = 1e-3, decay = 0.0), metrics=['accuracy'])
    return model

model = get_model()


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

batch_size = 32
epochs = 10
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_tra, X_val, y_tra, y_val = train_test_split(train_words, y_train, train_size=0.9, random_state=233)
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc,early_stop], verbose=1)


y_pred = model.predict(test_words, batch_size=1024)
          


# In[ ]:


submission = pd.read_csv('sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
submission.to_csv('bigru_preprocesing_v2.csv', index=False)           


# In[ ]:


submission

